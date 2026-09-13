import json
import tomllib

import pytest

from harness.core.artifacts import toml_dumps, write_result, write_run_result, write_task_toml
from harness.core.config import CaseConfig


def test_toml_roundtrip():
    doc = {
        "schema_version": "1.1", "seed": 3,
        "task": {"name": "x", "authors": [{"name": "a b", "email": "a@b"}]},
        "metadata": {"fail_to_pass": ["tests/t.py::a", "tests/t.py::b[1-2]"], "pass_to_pass": []},
        "limits": {"build_timeout_sec": 10, "flag": True},
        "environment": {"test_command": "sh /tests/test.sh", "note": 'quote " inside'},
    }
    assert tomllib.loads(toml_dumps(doc)) == doc


def test_task_toml_follows_protocol(demo_input, tmp_path):
    cfg = CaseConfig.load(demo_input)
    task = tmp_path / "task"
    task.mkdir()
    manifest = {"fail_to_pass": ["tests/t.py::f"], "pass_to_pass": ["tests/t.py::p"], "anti_cheat": []}
    write_task_toml(task, cfg, manifest, description="Исправить учёт возвратов", bank_domain="Расчёты")
    toml = tomllib.loads((task / "task.toml").read_text(encoding="utf-8"))
    assert toml["schema_version"] == "1.1"
    assert toml["task"] == {"name": cfg.case_id, "description": "Исправить учёт возвратов",
                            "authors": [{"name": cfg.author.name, "email": cfg.author.email}]}
    md = toml["metadata"]
    assert md["task_type"] == "agentic" and md["build_tool"] == "docker" and md["bank_domain"] == "Расчёты"
    assert md["language"] == cfg.language and md["difficulty"] == cfg.difficulty
    assert md["source"] == cfg.source and md["team"] == cfg.team
    assert md["fail_to_pass"] == ["tests/t.py::f"] and md["pass_to_pass"] == ["tests/t.py::p"] and md["anti_cheat"] == []
    assert toml["agent"] == {"timeout_sec": cfg.limits.agent_timeout_sec}
    assert toml["verifier"] == {"timeout_sec": cfg.limits.verifier_timeout_sec}
    assert toml["environment"] == {"allow_internet": False, "build_timeout_sec": cfg.limits.build_timeout_sec,
                                   "cpus": cfg.limits.cpus, "memory_mb": cfg.limits.memory_mb,
                                   "storage_mb": cfg.limits.storage_mb}


def test_result_json_follows_protocol(demo_input, tmp_path):
    cfg = CaseConfig.load(demo_input)
    out = tmp_path / "out"
    (out / "evidence").mkdir(parents=True)
    res = write_result(out, config=cfg, status="failed", error="boom", limitations=["x"], attempts=0, snapshot_sha256="")
    assert res["protocol_version"] == "1.0" and res["case_id"] == cfg.case_id
    assert res["task_path"] is None and res["evidence_path"] == "evidence"
    assert res["limitations"] == ["x", "boom"] and res["input_snapshot_sha256"] is None
    (out / "task").mkdir()
    (out / "task" / "task.toml").write_text("x = 1\n")
    res = write_result(out, config=cfg, status="ready", error=None, limitations=[], attempts=1, snapshot_sha256="abc")
    assert res["task_path"] == "task" and res["limitations"] == [] and res["input_snapshot_sha256"] == "abc"
    import pytest
    with pytest.raises(ValueError):
        write_result(out, config=cfg, status="unverified", error=None, limitations=[], attempts=0, snapshot_sha256="")
    res = write_result(out, config=cfg, case_id="x/y-r2", status="ready", error=None, limitations=[], attempts=1,
                       snapshot_sha256="abc")
    assert res["case_id"] == "x/y-r2"


def test_run_result_lists_every_case(demo_input, tmp_path):
    cfg = CaseConfig.load(demo_input)
    out = tmp_path / "out"
    (out / "evidence").mkdir(parents=True)
    cases = [{"requirement_id": "R1", "title": "a", "status": "ready", "attempts": 1, "error": None},
             {"requirement_id": "R3", "title": "b", "status": "case_failed", "attempts": 3, "error": "convergence failed: x"}]
    res = write_run_result(out, config=cfg, status="failed", error="1 of 2 case(s) failed", limitations=["l"],
                           snapshot_sha256="abc", cases=cases)
    assert res["task_path"] is None and res["cases_path"] == "cases" and res["evidence_path"] == "evidence"
    assert res["cases_ready"] == 1 and res["cases_failed"] == 1 and res["attempts"] == 4
    assert res["limitations"] == ["l", "case R3 (b) is case_failed: convergence failed: x", "1 of 2 case(s) failed"]
    assert json.loads((out / "result.json").read_text(encoding="utf-8"))["cases"] == cases
    with pytest.raises(ValueError):
        write_run_result(out, config=cfg, status="partial", error=None, limitations=[], snapshot_sha256="", cases=[])
