"""Per-case pipeline without Docker: a scripted verifier stands in for the sandbox, and the
multi-case run is exercised end to end with --skip-docker."""
import copy
import dataclasses
import json

import pytest

from harness.core.brief import parse_brief_spec
from harness.core.config import CaseConfig
from harness.core.llm.mock_client import MockLLMClient
from harness.core.pipeline import CASE_FAILED, CASE_READY, Pipeline, case_id_for, image_tag_for
from harness.core.synthesis import SynthesisEngine, materialize, parse_synthesis
from harness.localization.code_retriever import build_project_tree, localize, read_text
from harness.providers.python.detector import PythonStackDetector
from harness.providers.python.test_runner import CheckReport
from harness.validation.convergence import VerificationReport

F2P = "tests/test_netting_refunds.py::test_settled_refund_reduces_balance"
PROBLEMS_ORACLE = ["oracle: reward=0, expected 1", f"oracle: fail_to_pass test failed: {F2P}: AssertionError: 3 != 4"]
PROBLEMS_BASE = ["base: reward=1, expected 0", f"base: fail_to_pass test passed on original code: {F2P}"]


def _report(problems: list[str], *, build_ok: bool = True) -> VerificationReport:
    ok = not problems
    return VerificationReport(ok=ok, build_ok=build_ok, problems=list(problems),
                              base=CheckReport("base", True, reward=0), oracle=CheckReport("oracle", ok, reward=int(ok)),
                              logs={"oracle/pytest.log": "FAILED" if problems else ""})


class FakeVerifier:
    def __init__(self, reports: list[VerificationReport]) -> None:
        self.reports, self.calls = list(reports), []

    def verify(self, manifest, *, need_build):
        self.calls.append({"manifest": copy.deepcopy(manifest), "need_build": need_build})
        return self.reports.pop(0)


class FakeEnvBuilder:
    def build(self, **kwargs):
        pass


def _two_cases(mock_responses: dict) -> dict:
    """Demo brief plus a second bug requirement R3 with its own bundle (the invariant R2 stays)."""
    m = copy.deepcopy(mock_responses)
    m["analyze_brief"]["requirements"].append({
        "id": "R3", "title": "Rounding to cents", "statement": "net_balance rounds to 2 decimals",
        "acceptance_criteria": ["1.005 -> 1.01"], "kind": "bug", "testable": True})
    r3 = copy.deepcopy(m["synthesize"])
    r3["edits"] = [{"op": "create", "path": "ledger/rounding.py", "content": "PLACES = 2\n"}]
    r3["test_files"] = [{"path": "tests/test_rounding.py", "content":
                         "from ledger.models import Transaction\n\n\ndef test_r3():\n    assert 1\n\n\n"
                         "def test_unsettled_ignored():\n    assert 1\n\n\ndef test_dto():\n    assert 1\n"}]
    r3["fail_to_pass"] = ["tests/test_rounding.py::test_r3"]
    r3["pass_to_pass"] = ["tests/test_rounding.py::test_unsettled_ignored"]
    r3["anti_cheat"] = ["tests/test_rounding.py::test_dto"]
    r3["coverage"] = [{"requirement_id": "R3", "tests": ["tests/test_rounding.py::test_r3"]},
                      {"requirement_id": "R2", "tests": ["tests/test_rounding.py::test_unsettled_ignored"]}]
    r3["instruction_md"] = "# Округление баланса\n\n" + "Результат net_balance должен округляться до двух знаков. " * 4
    m["synthesize"] = [m["synthesize"], r3]
    return m


@pytest.fixture
def loop(demo_input, mock_responses, tmp_path):
    def make(responses: dict, *, max_retries: int):
        cfg = CaseConfig.load(demo_input)
        cfg = dataclasses.replace(cfg, limits=dataclasses.replace(cfg.limits, max_retries=max_retries))
        llm = MockLLMClient(responses)
        tree = build_project_tree(cfg.repository, untrusted_dirs=cfg.untrusted_dirs)
        profile = PythonStackDetector().detect(cfg.repository, tree)
        spec = parse_brief_spec(responses["analyze_brief"]).case_spec("R1")
        ctx = localize(tree, cfg.brief, spec=spec, index_dir=tmp_path / "idx", embedder=None, backend="keyword")
        read = lambda rel: read_text(cfg.repository / rel)  # noqa: E731
        engine = SynthesisEngine(llm, brief=cfg.brief, profile=profile, spec=spec, read_file=read)
        syn = parse_synthesis(responses["synthesize"], spec, read)
        pipeline = Pipeline(cfg, llm, search_backend="keyword", skip_docker=True)
        task_dir, evidence_dir = tmp_path / "case" / "task", tmp_path / "case" / "evidence"
        task_dir.mkdir(parents=True)
        evidence_dir.mkdir(parents=True)
        materialize(task_dir, syn)
        return pipeline, engine, ctx, syn, llm, task_dir, evidence_dir, tree, profile
    return make


def test_heal_then_converge(loop, mock_responses):
    responses = dict(mock_responses, heal=mock_responses["synthesize"])
    pipeline, engine, ctx, syn, llm, task_dir, evidence_dir, tree, profile = loop(responses, max_retries=2)
    verifier = FakeVerifier([_report(PROBLEMS_ORACLE), _report([])])
    report, attempts, syn = pipeline._converge(verifier, FakeEnvBuilder(), engine, ctx, syn, evidence_dir,
                                               task_dir=task_dir, tree=tree, profile=profile)
    assert report.ok and attempts == 2 and len(verifier.calls) == 2
    assert [p["purpose"] for p in llm.prompts] == ["heal"]
    assert "<verification_problems>" in llm.prompts[0]["user"] and F2P in llm.prompts[0]["user"]
    assert (evidence_dir / "attempts" / "01" / "task" / "solution" / "solve.sh").exists()


def test_unjustified_edit_change_is_rejected_before_the_sandbox(loop, mock_responses):
    bad = copy.deepcopy(mock_responses["synthesize"])
    bad["edits"][0]["new"] += "            pass\n"
    responses = dict(mock_responses, heal=[bad, mock_responses["synthesize"]])
    pipeline, engine, ctx, syn, llm, task_dir, evidence_dir, tree, profile = loop(responses, max_retries=2)
    verifier = FakeVerifier([_report(PROBLEMS_BASE), _report([])])
    report, attempts, syn = pipeline._converge(verifier, FakeEnvBuilder(), engine, ctx, syn, evidence_dir,
                                               task_dir=task_dir, tree=tree, profile=profile)
    assert report.ok and attempts == 3 and len(verifier.calls) == 2
    assert "TEST defects" in llm.prompts[1]["user"]
    assert syn.edits == parse_synthesis(mock_responses["synthesize"]).edits


def test_retries_exhausted_returns_failed_report(loop, mock_responses):
    responses = dict(mock_responses, heal=mock_responses["synthesize"])
    pipeline, engine, ctx, syn, llm, task_dir, evidence_dir, tree, profile = loop(responses, max_retries=1)
    verifier = FakeVerifier([_report(PROBLEMS_ORACLE), _report(PROBLEMS_ORACLE)])
    report, attempts, _ = pipeline._converge(verifier, FakeEnvBuilder(), engine, ctx, syn, evidence_dir,
                                             task_dir=task_dir, tree=tree, profile=profile)
    assert not report.ok and attempts == 2 and report.problems == PROBLEMS_ORACLE


def test_run_builds_one_case_per_requirement(demo_input, mock_responses):
    cfg = CaseConfig.load(demo_input)
    result = Pipeline(cfg, MockLLMClient(_two_cases(mock_responses)), search_backend="keyword", skip_docker=True).run()
    out = cfg.output_dir
    assert result["status"] == "failed" and result["error"] is None       # not verified: never ready
    assert result["task_path"] is None and result["cases_path"] == "cases" and result["evidence_path"] == "evidence"
    assert [c["requirement_id"] for c in result["cases"]] == ["R1", "R3"]
    assert all(c["status"] == CASE_FAILED and c["error"] is None for c in result["cases"])
    for rid, f2p in (("R1", "tests/test_netting_refunds.py"), ("R3", "tests/test_rounding.py")):
        case = out / "cases" / rid
        for rel in ("task/task.toml", "task/instruction.md", "task/solution/solve.sh", f"task/{f2p}",
                    "task/tests/test.sh", "task/environment/Dockerfile", "task/environment/repo/ledger/netting.py",
                    "evidence/case_spec.json", "evidence/localization.json", "evidence/synthesis.json",
                    "evidence/llm_usage.json", "evidence/summary.json", "result.json"):
            assert (case / rel).exists(), f"{rid}/{rel}"
        res = json.loads((case / "result.json").read_text(encoding="utf-8"))
        assert res["case_id"] == case_id_for(cfg.case_id, rid) and res["status"] == "failed"
        assert res["case_status"] == CASE_FAILED and res["requirement_id"] == rid
        assert res["task_path"] == "task" and res["input_snapshot_sha256"] == result["input_snapshot_sha256"]
        usage = json.loads((case / "evidence/llm_usage.json").read_text(encoding="utf-8"))
        assert usage["total_calls"] == 1 and usage["calls"][0]["purpose"] == "synthesize"
        cspec = json.loads((case / "evidence/case_spec.json").read_text(encoding="utf-8"))
        assert cspec["target"] == rid and [r["id"] for r in cspec["requirements"]] == [rid, "R2"]
    r3_task = json.loads((out / "cases/R3/evidence/synthesis.json").read_text(encoding="utf-8"))
    assert r3_task["edits"][0]["path"] == "ledger/rounding.py"
    assert (out / "evidence/brief_spec.json").exists() and (out / "evidence/llm_usage.json").exists()
    run_usage = json.loads((out / "evidence/llm_usage.json").read_text(encoding="utf-8"))
    assert run_usage["total_calls"] == 3
    assert not (out / "task").exists()
    assert sum("is case_failed: not verified" in lim for lim in result["limitations"]) == 2   # always listed


def test_failed_case_is_listed_and_the_others_continue(demo_input, mock_responses):
    responses = _two_cases(mock_responses)
    broken = copy.deepcopy(responses["synthesize"][0])
    broken["fail_to_pass"] = []            # validator rejects it every round -> SynthesisError
    responses["synthesize"] = [broken, responses["synthesize"][1]]
    responses["repair"] = broken
    cfg = CaseConfig.load(demo_input)
    result = Pipeline(cfg, MockLLMClient(responses), search_backend="keyword", skip_docker=True).run()
    r1, r3 = result["cases"]
    assert r1["status"] == CASE_FAILED and r1["failed_stage"] == "synthesis"
    assert "bundle rejected by the validator" in r1["error"] and "fail_to_pass is empty" in r1["error"]
    assert r3["status"] == CASE_FAILED and r3["error"] is None and (cfg.output_dir / "cases/R3/task/task.toml").exists()
    assert not (cfg.output_dir / "cases/R1/task/task.toml").exists()
    res1 = json.loads((cfg.output_dir / "cases/R1/result.json").read_text(encoding="utf-8"))
    assert res1["task_path"] is None and res1["error"] == r1["error"]
    assert any("case R1" in lim and "case_failed" in lim for lim in result["limitations"])
    assert result["cases_failed"] == 2 and result["error"].startswith("1 of 2 case(s) failed: R1")   # R3: no error, just unverified


def test_llm_failure_inside_a_case_is_explicit(demo_input, mock_responses):
    responses = _two_cases(mock_responses)
    responses["synthesize"] = [responses["synthesize"][1]]     # R1 gets R3's bundle -> invalid; then mock runs dry
    cfg = CaseConfig.load(demo_input)
    result = Pipeline(cfg, MockLLMClient({k: v for k, v in responses.items() if k != "synthesize"}),
                      search_backend="keyword", skip_docker=True).run()
    for c in result["cases"]:
        assert c["status"] == CASE_FAILED and c["error"].startswith("LLM step failed (synthesis)")
    assert result["status"] == "failed" and result["failed_stage"] == "6-packaging"


def test_no_case_when_brief_has_only_invariants(demo_input, mock_responses):
    responses = copy.deepcopy(mock_responses)
    for r in responses["analyze_brief"]["requirements"]:
        r["kind"] = "invariant"
    cfg = CaseConfig.load(demo_input)
    result = Pipeline(cfg, MockLLMClient(responses), search_backend="keyword", skip_docker=True).run()
    assert result["status"] == "failed" and result["cases"] == []
    assert "nothing to build a case from" in result["error"] and result["failed_stage"] == "2a-brief-analysis"


def test_ids_and_tags():
    assert case_id_for("team/name", "R2") == "team/name-r2"
    assert image_tag_for("team/name-r2", 7) == "harness-team-name-r2:7"
