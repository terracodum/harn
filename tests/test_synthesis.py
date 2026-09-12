import copy
import json

import pytest

from harness.core.synthesis import SynthesisError, materialize, parse_synthesis


def test_parse_demo_bundle(mock_responses):
    syn = parse_synthesis(mock_responses["synthesize"])
    assert "tests/test_netting_refunds.py" in syn.test_files
    assert len(syn.manifest["fail_to_pass"]) == 3
    assert syn.solve_sh.startswith("#!/bin/sh")


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


def test_empty_init_allowed(mock_responses):
    data = copy.deepcopy(mock_responses["synthesize"])
    data["test_files"].append({"path": "tests/__init__.py", "content": ""})
    syn = parse_synthesis(data)
    assert "tests/__init__.py" in syn.test_files


def test_bad_test_path_rejected(mock_responses):
    data = copy.deepcopy(mock_responses["synthesize"])
    data["test_files"][0]["path"] = "../evil.py"
    with pytest.raises(SynthesisError):
        parse_synthesis(data)


def test_materialize_writes_lf_files(tmp_path, mock_responses):
    syn = parse_synthesis(mock_responses["synthesize"])
    task = tmp_path / "task"
    materialize(task, syn)
    assert (task / "tests" / "test_netting_refunds.py").exists()
    assert b"\r\n" not in (task / "solution" / "solve.sh").read_bytes()
    manifest = json.loads((task / "tests" / "manifest.json").read_text())
    assert set(manifest) == {"fail_to_pass", "pass_to_pass", "anti_cheat"}
    assert (task / "instruction.md").read_text(encoding="utf-8").startswith("# ")
