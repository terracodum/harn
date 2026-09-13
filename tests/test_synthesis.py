import copy
import json

import pytest

from harness.core.brief import parse_brief_spec
from harness.core.synthesis import SynthesisError, instruction_problems, materialize, parse_synthesis


def test_parse_demo_bundle(mock_responses):
    syn = parse_synthesis(mock_responses["synthesize"])
    assert "tests/test_netting_refunds.py" in syn.test_files
    assert len(syn.manifest["fail_to_pass"]) == 3
    assert syn.solution.order == ["R1"] and syn.target == "R1"
    assert syn.solve_sh.startswith("#!/bin/sh")
    assert syn.to_dict()["edits"][0]["op"] == "replace"
    assert parse_synthesis(mock_responses["synthesize"], target="R4").target == "R4"


def test_uncategorised_test_rejected(mock_responses):
    data = copy.deepcopy(mock_responses["synthesize"])
    data["pass_to_pass"] = data["pass_to_pass"][:-1]
    with pytest.raises(SynthesisError, match="not categorised"):
        parse_synthesis(data)


def test_duplicate_category_rejected(mock_responses):
    data = copy.deepcopy(mock_responses["synthesize"])
    data["anti_cheat"].append(data["fail_to_pass"][0])
    with pytest.raises(SynthesisError, match="two categories"):
        parse_synthesis(data)


def test_spoiler_in_instruction_rejected(mock_responses):
    data = copy.deepcopy(mock_responses["synthesize"])
    data["instruction_md"] += "\nСм. solve.sh"
    with pytest.raises(SynthesisError, match="solve.sh"):
        parse_synthesis(data)
    assert instruction_problems("short") == ["instruction_md is too short"]
    assert instruction_problems("x" * 100 + " see solve_R2.sh") == ["instruction_md mentions 'solve_r'"]


def test_empty_init_allowed(mock_responses):
    data = copy.deepcopy(mock_responses["synthesize"])
    data["test_files"].append({"path": "tests/__init__.py", "content": ""})
    syn = parse_synthesis(data)
    assert "tests/__init__.py" in syn.test_files


def test_async_test_without_plugin_rejected(mock_responses):
    data = copy.deepcopy(mock_responses["synthesize"])
    data["test_files"][0]["content"] += "\n\nasync def test_async_thing():\n    assert True\n"
    data["pass_to_pass"].append("tests/test_netting_refunds.py::test_async_thing")
    with pytest.raises(SynthesisError, match="asyncio.run"):
        parse_synthesis(data)
    data["extra_pip_packages"] = ["pytest-asyncio"]
    parse_synthesis(data)


def test_bad_test_path_rejected(mock_responses):
    data = copy.deepcopy(mock_responses["synthesize"])
    data["test_files"][0]["path"] = "../evil.py"
    with pytest.raises(SynthesisError):
        parse_synthesis(data)


def test_case_bundle_must_cover_target_and_invariants(mock_responses):
    spec = parse_brief_spec(mock_responses["analyze_brief"]).case_spec("R1")      # R1 bug (target), R2 invariant
    data = copy.deepcopy(mock_responses["synthesize"])
    syn = parse_synthesis(data, spec)
    assert syn.target == "R1"
    data["coverage"] = [c for c in data["coverage"] if c["requirement_id"] != "R2"]
    with pytest.raises(SynthesisError, match="requirement R2 .* has no tests"):
        parse_synthesis(data, spec)
    data = copy.deepcopy(mock_responses["synthesize"])
    data["edits"] = []
    with pytest.raises(SynthesisError, match="edits must be a non-empty list"):
        parse_synthesis(data, spec)
    data["edits"] = [{"op": "replace", "path": "ledger/netting.py", "old": "a", "new": "a"}]
    with pytest.raises(SynthesisError, match="identical"):
        parse_synthesis(data, spec)


def test_edits_are_checked_against_the_repository(mock_responses, demo_repo_copy):
    from harness.localization.code_retriever import read_text
    read = lambda rel: read_text(demo_repo_copy / rel)  # noqa: E731
    data = copy.deepcopy(mock_responses["synthesize"])
    parse_synthesis(data, read_file=read)
    data["edits"][0]["old"] = "total += tx.amount  # not what the file says"
    with pytest.raises(SynthesisError, match="not found in the file as left by the previous steps"):
        parse_synthesis(data, read_file=read)
    parse_synthesis(data)                 # without a reader only the structure is validated


def test_materialize_writes_step_chain(tmp_path, mock_responses):
    syn = parse_synthesis(mock_responses["synthesize"])
    task = tmp_path / "task"
    materialize(task, syn)
    assert (task / "tests" / "test_netting_refunds.py").exists()
    solution = task / "solution"
    assert {p.name for p in solution.glob("*.sh")} == {"solve.sh", "solve_R1.sh"}
    assert b"\r\n" not in (solution / "solve.sh").read_bytes()
    solve = (solution / "solve.sh").read_text(encoding="utf-8")
    assert "[solve] step R1" in solve and "solve_R1.sh" not in solve      # self-contained orchestrator
    manifest = json.loads((task / "tests" / "manifest.json").read_text())
    assert set(manifest) == {"fail_to_pass", "pass_to_pass", "anti_cheat"}
    assert (task / "instruction.md").read_text(encoding="utf-8").startswith("# ")
