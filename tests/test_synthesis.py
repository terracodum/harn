import copy
import json

import pytest

from harness.core.brief import parse_brief_spec
from harness.core.synthesis import SynthesisError, instruction_problems, materialize, parse_synthesis


def test_parse_demo_bundle(mock_responses):
    syn = parse_synthesis(mock_responses["synthesize"])
    assert "tests/test_netting_refunds.py" in syn.test_files
    assert len(syn.manifest["fail_to_pass"]) == 3
    assert syn.solution.order == ["R1"]
    assert syn.solve_sh.startswith("#!/bin/sh")
    assert syn.to_dict()["solve_steps"][0]["requirement_id"] == "R1"


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


def test_solve_steps_must_match_the_spec(mock_responses):
    spec = parse_brief_spec(mock_responses["analyze_brief"])      # R1 bug, R2 invariant
    data = copy.deepcopy(mock_responses["synthesize"])
    parse_synthesis(data, spec)
    data["solve_steps"].append({"requirement_id": "R2", "edits": [{"op": "create", "path": "x.py", "content": "1"}]})
    with pytest.raises(SynthesisError, match="R2 is invariant and must not have a step"):
        parse_synthesis(data, spec)
    data["solve_steps"] = [{"requirement_id": "R7", "edits": [{"op": "create", "path": "x.py", "content": "1"}]}]
    with pytest.raises(SynthesisError) as exc:
        parse_synthesis(data, spec)
    assert "R1 (bug) has no step" in str(exc.value) and "unknown requirement R7" in str(exc.value)
    data["solve_steps"] = []
    with pytest.raises(SynthesisError, match="solve_steps is empty"):
        parse_synthesis(data, spec)


def test_edits_are_checked_against_the_repository(mock_responses, demo_repo_copy):
    from harness.localization.code_retriever import read_text
    read = lambda rel: read_text(demo_repo_copy / rel)  # noqa: E731
    data = copy.deepcopy(mock_responses["synthesize"])
    parse_synthesis(data, read_file=read)
    data["solve_steps"][0]["edits"][0]["old"] = "total += tx.amount  # not what the file says"
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


def test_prune_requirement_removes_step_tests_and_orphan_files(mock_responses):
    data = copy.deepcopy(mock_responses["synthesize"])
    data["solve_steps"].append({"requirement_id": "R3", "edits": [{"op": "create", "path": "ledger/r3.py", "content": "X = 1\n"}]})
    data["test_files"].append({"path": "tests/test_r3.py", "content": "def test_r3():\n    assert 1\n"})
    data["fail_to_pass"].append("tests/test_r3.py::test_r3")
    data["coverage"].append({"requirement_id": "R3", "tests": ["tests/test_r3.py::test_r3",
                                                                "tests/test_netting_refunds.py::test_unsettled_operations_are_ignored"]})
    syn = parse_synthesis(data)
    removed = syn.prune_requirement("R3", "never converged")
    assert removed == ["tests/test_r3.py::test_r3"]          # the shared test stays: R2 still uses it
    assert syn.solution.order == ["R1"] and syn.solution.pruned == {"R3": "never converged"}
    assert "tests/test_r3.py" not in syn.test_files
    assert "tests/test_r3.py::test_r3" not in syn.manifest["fail_to_pass"]
    assert "tests/test_netting_refunds.py::test_unsettled_operations_are_ignored" in syn.manifest["pass_to_pass"]
    assert {c["requirement_id"] for c in syn.coverage} == {"R1", "R2"}
