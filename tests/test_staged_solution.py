import shutil
import subprocess

import pytest

from harness.core.staged_solution import StagedSolution, failing_step_from_log, step_file


def _sol() -> StagedSolution:
    sol, problems = StagedSolution.from_items([
        {"requirement_id": "R3", "script": "echo three\n"},
        {"requirement_id": "R1", "script": "#!/bin/sh\nset -eu\necho one\n"},
    ])
    assert problems == []
    return sol


def test_steps_are_ordered_and_normalised():
    sol = _sol()
    assert sol.order == ["R1", "R3"]
    assert sol.steps["R3"].startswith("#!/bin/sh\nset -eu\n") and sol.steps["R3"].endswith("\n")
    assert sol.prefix("R1").order == ["R1"]
    assert "solve_R1.sh" in sol.orchestrator() and sol.orchestrator().index("R1") < sol.orchestrator().index("R3")
    assert sol.combined().startswith("#!/bin/sh\nset -eu\n") and "# ---- R3" in sol.combined()


def test_from_items_reports_every_problem():
    _, problems = StagedSolution.from_items([
        {"requirement_id": "X", "script": "echo"},
        {"requirement_id": "R1", "script": ""},
        {"requirement_id": "R2", "script": "cp /tests/x ."},
        {"requirement_id": "R2", "script": "echo"},
        "not an object",
    ])
    assert len(problems) == 5
    assert any("must look like R1" in p for p in problems)
    assert any("R1 is empty" in p for p in problems)
    assert any("/tests" in p for p in problems)
    assert any("two steps" in p for p in problems)


def test_prune_and_change_detection():
    a, b = _sol(), _sol()
    assert a.changed_steps(b) == []
    b.steps["R3"] = "#!/bin/sh\nset -eu\necho THREE\n"
    assert a.changed_steps(b) == ["R3"]
    assert a.prune("R3", "no convergence") is True
    assert a.order == ["R1"] and a.pruned == {"R3": "no convergence"}
    assert a.prune("R3", "again") is False
    assert a.changed_steps(b) == ["R3"]      # removed on one side counts as changed


def test_write_removes_stale_step_files(tmp_path):
    sol = _sol()
    sol.write(tmp_path)
    assert {p.name for p in tmp_path.glob("*.sh")} == {"solve.sh", "solve_R1.sh", "solve_R3.sh"}
    sol.prune("R3", "x")
    sol.write(tmp_path)
    assert {p.name for p in tmp_path.glob("*.sh")} == {"solve.sh", "solve_R1.sh"}
    assert b"\r\n" not in (tmp_path / "solve.sh").read_bytes()


def test_failing_step_from_log():
    assert failing_step_from_log("[solve] step R1\n[solve] step R2\nTraceback...\n") == "R2"
    assert failing_step_from_log("[solve] step R1\n[solve] done\n") is None
    assert failing_step_from_log("") is None


def test_orchestrator_runs_steps_in_order_and_stops_on_failure(tmp_path):
    sh = shutil.which("sh")
    if not sh:
        pytest.skip("no POSIX sh available")
    sol, _ = StagedSolution.from_items([
        {"requirement_id": "R1", "script": "echo one > out.txt\n"},
        {"requirement_id": "R2", "script": "echo two >> out.txt\nexit 3\n"},
        {"requirement_id": "R3", "script": "echo three >> out.txt\n"},
    ])
    sol.write(tmp_path / "solution")
    work = tmp_path / "work"
    work.mkdir()
    proc = subprocess.run([sh, str(tmp_path / "solution" / "solve.sh")], cwd=work, capture_output=True, text=True)
    assert proc.returncode == 3
    assert (work / "out.txt").read_text().split() == ["one", "two"]
    assert failing_step_from_log(proc.stdout) == "R2"
    assert step_file("R2") == "solve_R2.sh"
