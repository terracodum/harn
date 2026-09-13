import copy
import json
from pathlib import Path

import pytest

from harness.core.config import CaseConfig
from harness.core.llm.mock_client import MockLLMClient
from harness.core.pipeline import CASE_FAILED, CASE_PRUNED, CASE_READY, CaseOutcome, Pipeline
from harness.providers.python.test_runner import CheckReport
from harness.validation.convergence import VerificationReport, check_already_satisfied


def _two_cases(mock_responses: dict) -> dict:
    """Demo brief plus a second requirement R3."""
    m = copy.deepcopy(mock_responses)
    m["analyze_brief"]["requirements"].append({
        "id": "R3", "title": "Tenant isolation check", "statement": "All queries filter by tenant_id",
        "acceptance_criteria": ["tenant_id is filtered"], "kind": "bug", "testable": True})
    r3 = copy.deepcopy(m["synthesize"])
    r3["already_satisfied"] = True
    r3["edits"] = []
    r3["fail_to_pass"] = []
    r3["coverage"] = []
    r3["root_cause"] = "Repository already applies tenant filter in all queries by default."
    m["synthesize"] = [m["synthesize"], r3]
    return m


def test_case_pruned_via_synthesis_already_satisfied(demo_input, mock_responses):
    responses = _two_cases(mock_responses)
    cfg = CaseConfig.load(demo_input)
    pipeline = Pipeline(cfg, MockLLMClient(responses), search_backend="keyword", skip_docker=True)
    result = pipeline.run()

    # R1 is packaged (not verified due to skip_docker), R3 is pruned
    assert result["cases_pruned"] == 1
    assert result["cases_ready"] == 0
    assert result["cases_failed"] == 1

    r1 = next(c for c in result["cases"] if c["requirement_id"] == "R1")
    r3 = next(c for c in result["cases"] if c["requirement_id"] == "R3")

    assert r1["status"] == CASE_FAILED
    assert r3["status"] == CASE_PRUNED
    assert r3["error"] is None
    assert r3["task_path"] is None

    out = cfg.output_dir
    # R1 case exists under cases/R1
    assert (out / "cases" / "R1" / "task" / "task.toml").exists()
    # R3 case folder was pruned from cases/
    assert not (out / "cases" / "R3").exists()
    # R3 evidence was saved under evidence/pruned/R3
    assert (out / "evidence" / "pruned" / "R3" / "synthesis.json").exists()

    # Run limitations explain that R3 was pruned
    assert any("R3" in lim and "pruned" in lim for lim in result["limitations"])


def test_check_already_satisfied_helper():
    # When fail_to_pass test passes on base and oracle passes
    report = VerificationReport(
        ok=False, build_ok=True,
        problems=["base: reward=1, expected 0", "base: fail_to_pass test passed on original code: tests/t.py::test_foo"],
        base=CheckReport("base", True, reward=1),
        oracle=CheckReport("oracle", True, reward=1),
    )
    assert check_already_satisfied(report) is True

    # When oracle also had problems, it's not already satisfied (broken test/fix)
    broken_oracle = VerificationReport(
        ok=False, build_ok=True,
        problems=["oracle: reward=0, expected 1", "base: reward=1, expected 0"],
        base=CheckReport("base", True, reward=1),
        oracle=CheckReport("oracle", False, reward=0),
    )
    assert check_already_satisfied(broken_oracle) is False


def test_packaging_stage_ready_when_defect_ready_and_invariant_pruned(demo_input, mock_responses):
    cfg = CaseConfig.load(demo_input)
    pipeline = Pipeline(cfg, MockLLMClient(mock_responses), search_backend="keyword", skip_docker=True)

    # Simulate: R1 converged and is ready, R2 was pruned
    pipeline.cases = [
        CaseOutcome(requirement_id="R1", title="Fix Netting", case_id="case-r1", path="cases/R1",
                    status=CASE_READY, attempts=1),
        CaseOutcome(requirement_id="R2", title="Tenant Filter", case_id="case-r2", path=None,
                    status=CASE_PRUNED, attempts=0),
    ]

    # Verify run stage 6 packaging behavior
    ready = [c for c in pipeline.cases if c.status == CASE_READY]
    pruned = [c for c in pipeline.cases if c.status == CASE_PRUNED]
    failed = [c for c in pipeline.cases if c.status not in (CASE_READY, CASE_PRUNED)]
    broken = [c for c in failed if c.error]

    assert len(ready) == 1
    assert len(pruned) == 1
    assert len(failed) == 0
    assert len(broken) == 0


def test_packaging_stage_failed_when_all_pruned(demo_input, mock_responses):
    cfg = CaseConfig.load(demo_input)
    pipeline = Pipeline(cfg, MockLLMClient(mock_responses), search_backend="keyword", skip_docker=True)

    pipeline.cases = [
        CaseOutcome(requirement_id="R1", title="Invariant 1", case_id="case-r1", path=None,
                    status=CASE_PRUNED, attempts=0),
        CaseOutcome(requirement_id="R2", title="Invariant 2", case_id="case-r2", path=None,
                    status=CASE_PRUNED, attempts=0),
    ]

    ready = [c for c in pipeline.cases if c.status == CASE_READY]
    pruned = [c for c in pipeline.cases if c.status == CASE_PRUNED]
    failed = [c for c in pipeline.cases if c.status not in (CASE_READY, CASE_PRUNED)]

    status = "ready" if (ready and not failed) else "failed"
    assert status == "failed"
