import shutil
import subprocess

import pytest

from harness.core.staged_solution import StagedSolution, apply_edits, failing_step_from_log, parse_edits, step_file

REPO = {
    "pkg/a.py": "x = 1\ny = 2\nz = 3\n",
    "pkg/b.py": "def f():\n    return 1\n",
}


def read(path: str):
    return REPO.get(path)


def _items():
    return [
        {"requirement_id": "R3", "edits": [{"op": "replace", "path": "pkg/a.py", "old": "y = 20\n", "new": "y = 200\n"}]},
        {"requirement_id": "R1", "edits": [{"op": "replace", "path": "pkg/a.py", "old": "y = 2\n", "new": "y = 20\n"},
                                           {"op": "create", "path": "pkg/c.py", "content": "C = 1\n"}]},
    ]


def _sol() -> StagedSolution:
    sol, problems = StagedSolution.from_items(_items())
    assert problems == []
    return sol


def test_steps_are_ordered_and_apply_in_chain():
    sol = _sol()
    assert sol.order == ["R1", "R3"]
    assert sol.validate(read) == []
    state, problems = sol.state_after(read)
    assert problems == [] and state["pkg/a.py"] == "x = 1\ny = 200\nz = 3\n" and state["pkg/c.py"] == "C = 1\n"
    state1, _ = sol.state_after(read, upto="R1")
    assert state1["pkg/a.py"] == "x = 1\ny = 20\nz = 3\n"
    assert sol.prefix("R1").order == ["R1"]
    assert sol.touched_paths() == ["pkg/a.py", "pkg/c.py"]


def test_chain_validation_reports_where_an_edit_stops_applying():
    sol = _sol()
    problems = sol.state_after(read, steps=["R3"])[1]       # R3 without R1: `y = 20` does not exist yet
    assert len(problems) == 1 and problems[0].startswith("solve_steps R3 edit #1") and "not found" in problems[0]
    assert sol.dependents("R1", read) == ["R3"]
    assert sol.dependents("R3", read) == []
    sol2, _ = StagedSolution.from_items([{"requirement_id": "R1", "edits": [
        {"op": "replace", "path": "pkg/a.py", "old": "\n", "new": "\n\n"},
        {"op": "create", "path": "pkg/b.py", "content": ""},
        {"op": "delete", "path": "pkg/nope.py"}]}])
    problems = sol2.validate(read)
    assert any("found 3 times" in p for p in problems)
    assert any("already exists" in p for p in problems)
    assert any("does not exist" in p for p in problems)


def test_parse_edits_reports_every_problem():
    _, problems = parse_edits([
        {"op": "patch", "path": "pkg/a.py"},
        {"op": "replace", "path": "../evil.py", "old": "a", "new": "b"},
        {"op": "replace", "path": "tests/test_x.py", "old": "a", "new": "b"},
        {"op": "replace", "path": "pkg/a.py", "old": "", "new": "b"},
        {"op": "replace", "path": "pkg/a.py", "old": "same", "new": "same"},
        {"op": "create", "path": "pkg/n.py"},
        "nope",
    ], "R1")
    assert len(problems) == 7
    assert parse_edits([], "R1")[1] == ["R1: edits must be a non-empty list"]
    ok = [{"op": "create", "path": "pkg/n.py", "content": ""}]
    _, problems = StagedSolution.from_items([{"requirement_id": "X", "edits": ok}, {"requirement_id": "R1", "edits": ok},
                                             {"requirement_id": "R1", "edits": ok}, "nope"])
    assert any("must look like R1" in p for p in problems) and any("two steps" in p for p in problems)


def test_prune_and_change_detection():
    a, b = _sol(), _sol()
    assert a.changed_steps(b) == []
    b.steps["R3"] = [b.steps["R3"][0].__class__("replace", "pkg/a.py", old="y = 20\n", new="y = 2000\n")]
    assert a.changed_steps(b) == ["R3"]
    assert a.prune("R3", "no convergence") is True
    assert a.order == ["R1"] and a.pruned == {"R3": "no convergence"}
    assert a.prune("R3", "again") is False
    assert a.changed_steps(b) == ["R3"]      # removed on one side counts as changed
    assert a.to_items()[0]["edits"][0] == {"op": "replace", "path": "pkg/a.py", "old": "y = 2\n", "new": "y = 20\n"}


def test_write_renders_self_contained_solve_sh(tmp_path):
    sol = _sol()
    sol.write(tmp_path)
    assert {p.name for p in tmp_path.glob("*.sh")} == {"solve.sh", "solve_R1.sh", "solve_R3.sh"}
    solve = (tmp_path / "solve.sh").read_text(encoding="utf-8")
    assert "solve_R1.sh" not in solve                       # self-contained: no dependency on sibling files
    assert solve.index('[solve] step R1') < solve.index('[solve] step R3') < solve.index("[solve] done")
    assert "'old': 'y = 2\\n'" in solve and "REPO_PATH" in solve
    assert b"\r\n" not in (tmp_path / "solve.sh").read_bytes()
    sol.prune("R3", "x")
    sol.write(tmp_path)
    assert {p.name for p in tmp_path.glob("*.sh")} == {"solve.sh", "solve_R1.sh"}


def test_failing_step_from_log():
    assert failing_step_from_log("[solve] step R1\n[solve] step R2\nTraceback...\n") == "R2"
    assert failing_step_from_log("[solve] step R1\n[solve] done\n") is None
    assert failing_step_from_log("") is None


def test_rendered_scripts_apply_edits_and_stop_on_failure(tmp_path):
    sh = shutil.which("sh")
    if not sh:
        pytest.skip("no POSIX sh available")
    work = tmp_path / "repo"
    for path, text in REPO.items():
        (work / path).parent.mkdir(parents=True, exist_ok=True)
        (work / path).write_text(text, encoding="utf-8")
    sol, _ = StagedSolution.from_items(_items() + [
        {"requirement_id": "R4", "edits": [{"op": "replace", "path": "pkg/a.py", "old": "MISSING", "new": "x"}]},
        {"requirement_id": "R5", "edits": [{"op": "delete", "path": "pkg/b.py"}]},
    ])
    sol.write(tmp_path / "solution")
    proc = subprocess.run([sh, str(tmp_path / "solution" / "solve.sh")], cwd=work, capture_output=True, text=True)
    assert proc.returncode != 0
    assert (work / "pkg/a.py").read_text() == "x = 1\ny = 200\nz = 3\n"      # R1 and R3 applied
    assert (work / "pkg/c.py").exists() and (work / "pkg/b.py").exists()     # R5 never ran
    assert failing_step_from_log(proc.stdout) == "R4" and "R4 edit 1" in proc.stderr + proc.stdout
    # a single step file works on its own too (used by the staged diagnostics)
    proc = subprocess.run([sh, str(tmp_path / "solution" / step_file("R5"))], cwd=work, capture_output=True, text=True)
    assert proc.returncode == 0 and not (work / "pkg/b.py").exists()


def test_apply_edits_is_lazy_and_incremental():
    state: dict = {}
    edits, _ = parse_edits([{"op": "replace", "path": "pkg/b.py", "old": "return 1", "new": "return 2"}], "R1")
    assert apply_edits(edits, state, read, "R1") == []
    assert state == {"pkg/b.py": "def f():\n    return 2\n"}
