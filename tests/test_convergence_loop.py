"""The healing loop without Docker: a scripted verifier stands in for the sandbox."""
import copy
import dataclasses
import json

import pytest

from harness.core.brief import parse_brief_spec
from harness.core.config import CaseConfig
from harness.core.llm.mock_client import MockLLMClient
from harness.core.pipeline import Pipeline, prune_candidates
from harness.core.synthesis import SynthesisEngine, materialize, parse_synthesis
from harness.localization.code_retriever import build_project_tree, localize
from harness.providers.python.detector import PythonStackDetector
from harness.providers.python.test_runner import CheckReport
from harness.validation.convergence import VerificationReport, attribute_problems

R3_TEST = "tests/test_r3.py::test_r3"
R3_PROBLEMS = ["oracle: reward=0, expected 1", f"oracle: fail_to_pass test failed: {R3_TEST}: AssertionError: 3 != 4"]


def _with_r3(mock_responses: dict) -> dict:
    """Demo bundle plus a second bug requirement R3 with its own step and test."""
    m = copy.deepcopy(mock_responses)
    m["analyze_brief"]["requirements"].append({
        "id": "R3", "title": "Rounding to cents", "statement": "net_balance rounds to 2 decimals",
        "acceptance_criteria": ["1.005 -> 1.01"], "kind": "bug", "testable": True})
    s = m["synthesize"]
    s["solve_steps"].append({"requirement_id": "R3", "script": "echo r3\n"})
    s["test_files"].append({"path": "tests/test_r3.py", "content": "def test_r3():\n    assert 3 == 4\n"})
    s["fail_to_pass"].append(R3_TEST)
    s["coverage"].append({"requirement_id": "R3", "tests": [R3_TEST]})
    s["instruction_md"] += "\n## Округление\nРезультат округляется до копеек.\n"
    m["rewrite_instruction"] = {"instruction_md": s["instruction_md"].replace(
        "\n## Округление\nРезультат округляется до копеек.\n", "")}
    return m


def _report(problems: list[str], *, build_ok: bool = True) -> VerificationReport:
    ok = not problems
    return VerificationReport(ok=ok, build_ok=build_ok, problems=list(problems),
                              base=CheckReport("base", True, reward=0), oracle=CheckReport("oracle", ok, reward=int(ok)),
                              logs={"oracle/pytest.log": "FAILED tests/test_r3.py::test_r3" if problems else ""})


class FakeVerifier:
    def __init__(self, reports: list[VerificationReport], staged: list[str] | None = None) -> None:
        self.reports, self.calls, self.staged = list(reports), [], staged or []

    def verify(self, manifest, *, need_build):
        self.calls.append({"manifest": copy.deepcopy(manifest), "need_build": need_build})
        return self.reports.pop(0)

    def locate_failing_step(self, report, solution, manifest, coverage):
        return list(self.staged)


class FakeEnvBuilder:
    def build(self, **kwargs):  # noqa: D401 - nothing to build without docker
        pass


@pytest.fixture
def loop(demo_input, mock_responses, tmp_path):
    def make(responses: dict, *, max_retries: int):
        cfg = CaseConfig.load(demo_input)
        cfg = dataclasses.replace(cfg, limits=dataclasses.replace(cfg.limits, max_retries=max_retries))
        llm = MockLLMClient(responses)
        tree = build_project_tree(cfg.repository, untrusted_dirs=cfg.untrusted_dirs)
        profile = PythonStackDetector().detect(cfg.repository, tree)
        spec = parse_brief_spec(responses["analyze_brief"])
        ctx = localize(tree, cfg.brief, spec=spec, index_dir=tmp_path / "idx", embedder=None, backend="keyword")
        engine = SynthesisEngine(llm, brief=cfg.brief, profile=profile, spec=spec)
        syn = parse_synthesis(responses["synthesize"], spec)
        pipeline = Pipeline(cfg, llm, search_backend="keyword", skip_docker=True)
        pipeline.output_dir.mkdir(parents=True)
        pipeline.task_dir.mkdir()
        pipeline.evidence_dir.mkdir()
        materialize(pipeline.task_dir, syn)
        return pipeline, engine, ctx, spec, syn, llm, tree, profile
    return make


def test_unconverged_requirement_is_pruned_and_instruction_rewritten(loop, mock_responses):
    responses = _with_r3(mock_responses)
    responses["heal"] = responses["synthesize"]          # the model keeps returning the same bundle
    pipeline, engine, ctx, spec, syn, llm, tree, profile = loop(responses, max_retries=1)
    verifier = FakeVerifier([_report(R3_PROBLEMS), _report(R3_PROBLEMS), _report([])])

    report, attempts, syn = pipeline._converge(verifier, FakeEnvBuilder(), engine, ctx, spec, syn, tree=tree, profile=profile)

    assert report.ok and attempts == 3
    assert syn.solution.order == ["R1"] and syn.solution.pruned == {"R3": syn.solution.pruned["R3"]}
    assert [r.id for r in spec.requirements] == ["R1", "R2"] and spec.pruned[0]["id"] == "R3"
    assert R3_TEST not in verifier.calls[-1]["manifest"]["fail_to_pass"]
    assert "Округление" not in syn.instruction_md and "solve" not in syn.instruction_md.lower()
    assert [p["purpose"] for p in llm.prompts] == ["heal", "rewrite_instruction"]
    heal_prompt = llm.prompts[0]["user"]
    assert "<frozen>\n- R1\n- R2\n</frozen>" in heal_prompt and "<focus>\n- R3:" in heal_prompt
    assert any("R3" in lim and "pruned" in lim for lim in pipeline.limitations)
    solution_dir = pipeline.task_dir / "solution"
    assert {p.name for p in solution_dir.glob("*.sh")} == {"solve.sh", "solve_R1.sh"}
    assert not (pipeline.task_dir / "tests" / "test_r3.py").exists()
    assert json.loads((pipeline.evidence_dir / "pruned.json").read_text(encoding="utf-8"))["requirements"][0]["id"] == "R3"
    assert json.loads((pipeline.evidence_dir / "brief_spec.json").read_text(encoding="utf-8"))["pruned"]


def test_healed_bundle_touching_a_frozen_step_is_rejected(loop, mock_responses):
    responses = _with_r3(mock_responses)
    bad = copy.deepcopy(responses["synthesize"])
    bad["solve_steps"][0]["script"] = "echo tampered with R1\n"          # R1 converged -> frozen
    responses["heal"] = [bad, responses["synthesize"]]
    pipeline, engine, ctx, spec, syn, llm, tree, profile = loop(responses, max_retries=2)
    verifier = FakeVerifier([_report(R3_PROBLEMS), _report([])])

    report, attempts, syn = pipeline._converge(verifier, FakeEnvBuilder(), engine, ctx, spec, syn, tree=tree, profile=profile)

    assert report.ok and attempts == 3 and len(verifier.calls) == 2
    assert syn.solution.steps["R1"] == parse_synthesis(responses["synthesize"], spec).solution.steps["R1"]
    second_heal = llm.prompts[1]["user"]
    assert "rejected before execution" in second_heal and "converged requirement(s) R1" in second_heal
    assert syn.solution.pruned == {}


def test_unjustified_solution_change_is_rejected(loop, mock_responses):
    responses = _with_r3(mock_responses)
    bad = copy.deepcopy(responses["synthesize"])
    bad["solve_steps"][1]["script"] = "echo bent R3 to satisfy a wrong test\n"
    responses["heal"] = [bad, responses["synthesize"]]
    pipeline, engine, ctx, spec, syn, llm, tree, profile = loop(responses, max_retries=2)
    base_side = ["base: reward=1, expected 0", f"base: fail_to_pass test passed on original code: {R3_TEST}"]
    verifier = FakeVerifier([_report(base_side), _report([])])

    report, attempts, syn = pipeline._converge(verifier, FakeEnvBuilder(), engine, ctx, spec, syn, tree=tree, profile=profile)

    assert report.ok
    assert "TEST defects" in llm.prompts[1]["user"]


def test_pruning_is_refused_when_problems_are_not_attributable(loop, mock_responses):
    responses = _with_r3(mock_responses)
    responses["heal"] = responses["synthesize"]
    pipeline, engine, ctx, spec, syn, llm, tree, profile = loop(responses, max_retries=0)
    verifier = FakeVerifier([_report(["isolated base/fail_to_pass: reward=1, expected 0"])])

    report, attempts, syn = pipeline._converge(verifier, FakeEnvBuilder(), engine, ctx, spec, syn, tree=tree, profile=profile)

    assert not report.ok and attempts == 1 and len(verifier.calls) == 1
    assert syn.solution.order == ["R1", "R3"] and spec.pruned == []
    assert any(lim.startswith("pruning not possible") for lim in pipeline.limitations)
    assert all(p["purpose"] != "rewrite_instruction" for p in llm.prompts)


def test_prune_candidates_keeps_the_case_meaningful(mock_responses):
    responses = _with_r3(mock_responses)
    spec = parse_brief_spec(responses["analyze_brief"])
    syn = parse_synthesis(responses["synthesize"], spec)
    only_r1_fails = attribute_problems(
        ["oracle: fail_to_pass test failed: tests/test_netting_refunds.py::test_settled_refund_reduces_balance: x"],
        syn.manifest, syn.coverage)
    assert prune_candidates(only_r1_fails, spec, syn) == (["R1"], None)
    everything_fails = attribute_problems(
        ["oracle: fail_to_pass test failed: tests/test_netting_refunds.py::test_settled_refund_reduces_balance: x",
         f"oracle: fail_to_pass test failed: {R3_TEST}: y"], syn.manifest, syn.coverage)
    ids, why = prune_candidates(everything_fails, spec, syn)
    assert ids == [] and "no bug/feature/change requirement" in why


def test_rewrite_instruction_failure_fails_the_case(loop, mock_responses):
    responses = _with_r3(mock_responses)
    responses["heal"] = responses["synthesize"]
    responses["rewrite_instruction"] = {"instruction_md": "x" * 100 + " смотри solve.sh"}     # spoiler, every round
    pipeline, engine, ctx, spec, syn, llm, tree, profile = loop(responses, max_retries=0)
    verifier = FakeVerifier([_report(R3_PROBLEMS)])
    from harness.core.synthesis import SynthesisError
    with pytest.raises(SynthesisError, match="rewritten instruction rejected"):
        pipeline._converge(verifier, FakeEnvBuilder(), engine, ctx, spec, syn, tree=tree, profile=profile)
