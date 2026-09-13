import copy

import pytest

from harness.core.brief import BriefSpec, Requirement, analyze_brief, coverage_problems, parse_brief_spec
from harness.core.errors import PipelineError
from harness.core.llm.mock_client import MockLLMClient
from harness.core.synthesis import SynthesisError, parse_synthesis

BRIEF = ("При суточном закрытии возвраты со статусом settled увеличивают баланс, а должны уменьшать. "
         "Операции без settled_at по-прежнему игнорируются. Публичные интерфейсы не менять.")


def test_heuristic_spec_only_in_explicit_no_llm_mode():
    spec = analyze_brief(None, BRIEF, ["ledger/netting.py"])
    assert spec.source == "heuristic"
    assert spec.brief_language == "ru"
    assert len(spec.requirements) == 1 and spec.requirements[0].id == "R1"
    assert any("не менять" in c for c in spec.constraints)
    assert "settled_at" in spec.entities
    assert spec.search_queries[0] == BRIEF


def test_llm_spec(mock_responses):
    spec = analyze_brief(MockLLMClient(mock_responses), BRIEF, ["ledger/netting.py"])
    assert spec.source == "llm"
    assert [r.id for r in spec.requirements] == ["R1", "R2"]
    assert spec.requirements[1].kind == "invariant"
    assert spec.candidate_files == ["ledger/netting.py", "ledger/models.py"]


def test_llm_failure_is_raised_not_degraded():
    llm = MockLLMClient({})     # no canned reply -> LLMError inside
    with pytest.raises(PipelineError, match="brief analysis failed after 2 attempt"):
        analyze_brief(llm, BRIEF, [], retries=1)
    assert len(llm.prompts) == 2    # bounded retry, then stop


def test_invalid_llm_reply_is_retried_then_raised(mock_responses):
    bad = copy.deepcopy(mock_responses["analyze_brief"])
    bad["requirements"][0]["kind"] = "weird"
    llm = MockLLMClient({"analyze_brief": [bad, mock_responses["analyze_brief"]]})
    spec = analyze_brief(llm, BRIEF, [], retries=1)
    assert spec.source == "llm" and len(llm.prompts) == 2
    llm = MockLLMClient({"analyze_brief": bad})
    with pytest.raises(PipelineError, match="unknown kind 'weird'"):
        analyze_brief(llm, BRIEF, [], retries=0)


def test_parse_brief_spec_is_strict():
    spec = parse_brief_spec({"summary": "s", "requirements": [
        {"id": "REQ-A", "title": "a", "statement": "must a", "acceptance_criteria": [], "kind": "bug", "testable": True},
        {"id": "", "title": "b", "statement": "must b", "acceptance_criteria": ["x"], "kind": "feature", "testable": False},
    ]})
    assert [r.id for r in spec.requirements] == ["R1", "R2"]
    assert spec.testable_ids == ["R1"]
    with pytest.raises(ValueError, match="no requirements"):
        parse_brief_spec({"requirements": []})
    with pytest.raises(ValueError, match="unknown kind 'weird'"):
        parse_brief_spec({"requirements": [{"statement": "x", "kind": "weird"}]})
    with pytest.raises(ValueError, match="#2 has an empty statement"):
        parse_brief_spec({"requirements": [{"statement": "x", "kind": "bug"}, {"statement": "", "kind": "bug"}]})


MANIFEST = {"fail_to_pass": ["tests/t.py::f"], "pass_to_pass": ["tests/t.py::p"], "anti_cheat": ["tests/t.py::a"]}


def _spec(kind: str, testable: bool = True) -> BriefSpec:
    return BriefSpec(summary="s", requirements=[Requirement("R1", "t", "s", kind=kind, testable=testable)])


def test_coverage_rules_follow_the_requirement_kind():
    assert coverage_problems(_spec("bug"), [{"requirement_id": "R1", "tests": ["tests/t.py::f"]}], MANIFEST) == []
    # a `bug` covered only by pass_to_pass is rejected: the kind is never re-labelled empirically
    assert any("no fail_to_pass" in p for p in
               coverage_problems(_spec("bug"), [{"requirement_id": "R1", "tests": ["tests/t.py::p"]}], MANIFEST))
    assert coverage_problems(_spec("invariant"), [{"requirement_id": "R1", "tests": ["tests/t.py::p"]}], MANIFEST) == []
    assert any("is an invariant" in p for p in
               coverage_problems(_spec("invariant"), [{"requirement_id": "R1", "tests": ["tests/t.py::f"]}], MANIFEST))
    assert coverage_problems(_spec("bug"), [], MANIFEST)
    assert coverage_problems(_spec("bug", testable=False), [], MANIFEST) == []
    assert any("unknown test" in p for p in
               coverage_problems(_spec("bug"), [{"requirement_id": "R1", "tests": ["tests/t.py::nope"]}], MANIFEST))
    assert any("unknown requirement" in p for p in
               coverage_problems(_spec("bug"), [{"requirement_id": "R1", "tests": ["tests/t.py::f"]},
                                                {"requirement_id": "R9", "tests": ["tests/t.py::p"]}], MANIFEST))


def test_spec_prune_records_the_requirement():
    spec = BriefSpec(summary="s", requirements=[Requirement("R1", "a", "s"), Requirement("R2", "b", "s")])
    assert spec.prune("R2", "never converged").id == "R2"
    assert [r.id for r in spec.requirements] == ["R1"]
    assert spec.pruned == [{"id": "R2", "title": "b", "reason": "never converged"}]
    assert spec.prune("R9", "x") is None


def test_synthesis_rejects_uncovered_requirement(mock_responses):
    spec = parse_brief_spec(mock_responses["analyze_brief"])
    data = copy.deepcopy(mock_responses["synthesize"])
    assert parse_synthesis(data, spec).coverage
    data["coverage"] = [c for c in data["coverage"] if c["requirement_id"] != "R1"]
    with pytest.raises(SynthesisError, match="R1"):
        parse_synthesis(data, spec)
