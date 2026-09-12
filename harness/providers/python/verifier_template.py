#!/usr/bin/env python3
"""Verifier for a benchmark case. Copied verbatim into task/tests/verify.py.

Runs inside the sandbox (stdlib only):
  python /tests/verify.py --category all --manifest /tests/manifest.json \
      --tests-dir /tests --logs-dir /logs/verifier

  * runs pytest for the manifest tests of the chosen category
    (all | fail_to_pass | pass_to_pass | anti_cheat), JUnit XML -> tests.xml
  * parses the XML; a test counts as passed only with outcome "passed"
    (skip / xfail / error / missing all count as not passed)
  * writes reward.txt ("1" iff every required test passed, else "0")
    and results.json with the per-test outcomes.

The host side of the harness imports `parse_junit` / `compute_reward` from
this same file, so the two sides can never disagree.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

CATEGORIES = ("fail_to_pass", "pass_to_pass", "anti_cheat")


def load_manifest(path: str | Path) -> dict[str, list[str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {c: list(data.get(c, [])) for c in CATEGORIES}


def select_ids(manifest: dict[str, list[str]], category: str) -> list[str]:
    if category == "all":
        return [t for c in CATEGORIES for t in manifest[c]]
    if category not in manifest:
        raise SystemExit(f"unknown category: {category}")
    return list(manifest[category])


def test_file_of(test_id: str) -> str:
    return test_id.split("::", 1)[0]


def to_container_nodeid(test_id: str, tests_dir: str) -> str:
    """'tests/test_x.py::test_y' -> '/tests/test_x.py::test_y'."""
    rel = test_id.split("/", 1)[1] if test_id.startswith("tests/") else test_id
    return f"{tests_dir.rstrip('/')}/{rel}"


def run_pytest(nodeids: list[str], *, xml_path: Path, log_path: Path, tests_dir: str, cwd: str,
               rootdir: str = "/") -> int:
    cmd = [sys.executable, "-m", "pytest", f"--rootdir={rootdir}", "-p", "no:cacheprovider", "-c", f"{tests_dir}/pytest.ini",
           "-v", "-rA", "--junitxml", str(xml_path), *nodeids]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        text = proc.stdout.decode("utf-8", errors="replace")
        log.write(text)
    sys.stdout.write(text)
    return proc.returncode


def parse_junit(xml_path: str | Path, known_files: set[str]) -> dict[str, dict[str, str]]:
    """JUnit XML -> {test_id: {"outcome": ..., "message": ...}}.

    `known_files` are the test files relative to the mount parent
    ("tests/test_x.py"); pytest's classname is "tests.test_x[.Class]".
    """
    results: dict[str, dict[str, str]] = {}
    xml_path = Path(xml_path)
    if not xml_path.exists():
        return results
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return results
    for case in root.iter("testcase"):
        classname = case.get("classname", "")
        name = case.get("name", "")
        parts = classname.split(".") if classname else []
        file_rel, classes = None, []
        for k in range(len(parts), 0, -1):
            cand = "/".join(parts[:k]) + ".py"
            if cand in known_files:
                file_rel, classes = cand, parts[k:]
                break
        if file_rel is None:
            file_rel = ("/".join(parts) + ".py") if parts else "unknown.py"
        test_id = "::".join([file_rel, *classes, name])
        outcome, message = "passed", ""
        for tag, out in (("error", "error"), ("failure", "failed"), ("skipped", "skipped")):
            node = case.find(tag)
            if node is not None:
                outcome = out
                message = (node.get("message") or "")[:2000] + "\n" + (node.text or "")[:4000]
                break
        results[test_id] = {"outcome": outcome, "message": message.strip()}
    for err in root.iter("error"):  # collection errors attached to <testsuite>
        if err not in [c.find("error") for c in root.iter("testcase")]:
            results.setdefault("__collection__", {"outcome": "error", "message": (err.get("message") or "")[:4000]})
    return results


def compute_reward(required: list[str], results: dict[str, dict[str, str]]) -> tuple[int, dict[str, dict[str, str]]]:
    table: dict[str, dict[str, str]] = {}
    ok = bool(required)
    for tid in required:
        row = results.get(tid) or {"outcome": "missing", "message": "not found in JUnit XML"}
        table[tid] = row
        if row["outcome"] != "passed":
            ok = False
    return (1 if ok else 0), table


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="all")
    ap.add_argument("--manifest", default="/tests/manifest.json")
    ap.add_argument("--tests-dir", default="/tests")
    ap.add_argument("--logs-dir", default="/logs/verifier")
    ap.add_argument("--cwd", default="/app/repo")
    ap.add_argument("--rootdir", default="/", help="pytest rootdir; must be the parent of --tests-dir")
    args = ap.parse_args(argv)

    logs = Path(args.logs_dir)
    logs.mkdir(parents=True, exist_ok=True)
    reward_path, xml_path, log_path = logs / "reward.txt", logs / "tests.xml", logs / "pytest.log"
    reward_path.write_text("0\n", encoding="utf-8")  # pessimistic default

    manifest = load_manifest(args.manifest)
    required = select_ids(manifest, args.category)
    known_files = {test_file_of(t) for c in CATEGORIES for t in manifest[c]}
    nodeids = sorted({to_container_nodeid(test_file_of(t), args.tests_dir) for t in required})

    exit_code = run_pytest(nodeids, xml_path=xml_path, log_path=log_path, tests_dir=args.tests_dir, cwd=args.cwd,
                           rootdir=args.rootdir) if nodeids else 5
    results = parse_junit(xml_path, known_files)
    reward, table = compute_reward(required, results)
    reward_path.write_text(f"{reward}\n", encoding="utf-8")
    (logs / "results.json").write_text(json.dumps({
        "category": args.category,
        "required": required,
        "pytest_exit_code": exit_code,
        "reward": reward,
        "tests": table,
        "all_tests": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[verify] category={args.category} required={len(required)} reward={reward}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
