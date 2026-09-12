import copy

import pytest

from harness.core.brief import BriefSpec, Requirement, analyze_brief, coverage_problems, parse_brief_spec
from harness.core.llm.base_client import LLMError
from harness.core.llm.mock_client import MockLLMClient
from harness.core.synthesis import SynthesisError, parse_synthesis

BRIEF = ("При суточном закрытии возвраты со статусом settled увеличивают баланс, а должны уменьшать. "
         "Операции без settled_at по-прежнему игнорируются. Публичные интерфейсы не менять.")


def test_heuristic_spec_without_llm():
    spec = analyze_brief(None, BRIEF, ["ledger/netting.py"])
    assert spec.source == "heuristic"
    assert spec.brief_language == "ru"
    assert len(spec.requirements) == 1 and spec.requirements[0].id == "R1"
    assert any("не менять" in c for c in spec.constraints)
    assert "settled_at" in spec.entities
    assert spec.search_queries[0] == BRIEF


def test_llm_spec_and_fallback(mock_responses):
    spec = analyze_brief(MockLLMClient(mock_responses), BRIEF, ["ledger/netting.py"])
    assert spec.source == "llm"
    assert [r.id for r in spec.requirements] == ["R1", "R2"]
    assert spec.requirements[1].kind == "invariant"
    assert spec.candidate_files == ["ledger/netting.py", "ledger/models.py"]
    # model failure degrades to heuristics instead of stopping the run
    spec2 = analyze_brief(MockLLMClient({}), BRIEF, [])
    assert spec2.source == "heuristic" and spec2.ambiguities


def test_parse_brief_spec_normalises_ids():
    spec = parse_brief_spec({"summary": "s", "requirements": [
        {"id": "REQ-A", "title": "a", "statement": "must a", "acceptance_criteria": [], "kind": "weird", "testable": True},
        {"id": "", "title": "b", "statement": "must b", "acceptance_criteria": ["x"], "kind": "feature", "testable": False},
    ]})
    assert [r.id for r in spec.requirements] == ["R1", "R2"]
    assert spec.requirements[0].kind == "bug"
    assert spec.testable_ids == ["R1"]
    with pytest.raises(ValueError):
        parse_brief_spec({"requirements": []})


MANIFEST = {"fail_to_pass": ["tests/t.py::f"], "pass_to_pass": ["tests/t.py::p"], "anti_cheat": ["tests/t.py::a"]}


def _spec(kind: str, testable: bool = True) -> BriefSpec:
    return BriefSpec(summary="s", requirements=[Requirement("R1", "t", "s", kind=kind, testable=testable)])


def test_coverage_rules():
    assert coverage_problems(_spec("bug"), [{"requirement_id": "R1", "tests": ["tests/t.py::f"]}], MANIFEST) == []
    # a `bug` requirement covered only by pass_to_pass is accepted here: the Base run decides empirically
    assert coverage_problems(_spec("bug"), [{"requirement_id": "R1", "tests": ["tests/t.py::p"]}], MANIFEST) == []
    assert coverage_problems(_spec("invariant"), [{"requirement_id": "R1", "tests": ["tests/t.py::p"]}], MANIFEST) == []
    assert coverage_problems(_spec("bug"), [], MANIFEST)
    assert coverage_problems(_spec("bug", testable=False), [], MANIFEST) == []
    assert any("unknown test" in p for p in
               coverage_problems(_spec("bug"), [{"requirement_id": "R1", "tests": ["tests/t.py::nope"]}], MANIFEST))


def test_reclassify_from_base_run():
    from harness.core.brief import reclassify_from_base_run
    spec = BriefSpec(summary="s", requirements=[Requirement("R1", "real bug", "s", kind="bug"),
                                                Requirement("R2", "already fine", "s", kind="bug")])
    coverage = [{"requirement_id": "R1", "tests": ["tests/t.py::a"]}, {"requirement_id": "R2", "tests": ["tests/t.py::b"]}]
    notes = reclassify_from_base_run(spec, coverage, {"tests/t.py::a": "failed", "tests/t.py::b": "passed"})
    assert spec.requirements[0].kind == "bug" and spec.requirements[1].kind == "invariant"
    assert len(notes) == 1 and "R2" in notes[0]


def test_synthesis_rejects_uncovered_requirement(mock_responses):
    spec = parse_brief_spec(mock_responses["analyze_brief"])
    data = copy.deepcopy(mock_responses["synthesize"])
    assert parse_synthesis(data, spec).coverage
    data["coverage"] = [c for c in data["coverage"] if c["requirement_id"] != "R1"]
    with pytest.raises(SynthesisError, match="R1"):
        parse_synthesis(data, spec)
