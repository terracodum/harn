"""Host-side reading of a sandbox run and checking it against expectations."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from harness.providers.base import ITestRunner, RunResult, TestStatus
from harness.providers.python.verifier_template import CATEGORIES, compute_reward, parse_junit, test_file_of

MSG_LIMIT = 1500  # chars of a failure message forwarded to the healing prompt
_ENV_FAILURE = re.compile(r"SyntaxError|IndentationError|No module named 'pytest'|fixture '.*' not found", re.IGNORECASE)


class PytestJUnitRunner(ITestRunner):
    def parse_results(self, logs_dir: Path, manifest: dict[str, list[str]], category: str) -> RunResult:
        verifier = logs_dir / "verifier"
        reward: int | None = None
        reward_file = verifier / "reward.txt"
        if reward_file.exists():
            txt = reward_file.read_text(encoding="utf-8").strip()
            reward = int(txt) if txt in ("0", "1") else None

        required = [t for c in CATEGORIES for t in manifest[c]] if category == "all" else list(manifest[category])
        results_json = verifier / "results.json"
        if results_json.exists():
            data = json.loads(results_json.read_text(encoding="utf-8"))
            raw = data.get("all_tests") or data.get("tests") or {}
        else:
            known = {test_file_of(t) for c in CATEGORIES for t in manifest[c]}
            raw = parse_junit(verifier / "tests.xml", known)
        _, table = compute_reward(required, raw)
        tests = {tid: TestStatus(tid, row["outcome"], row.get("message", "")) for tid, row in table.items()}
        return RunResult(reward=reward, tests=tests, exit_code=None, duration_sec=0.0)


@dataclass
class CheckReport:
    kind: str                       # base | oracle
    ok: bool
    problems: list[str] = field(default_factory=list)
    reward: int | None = None
    table: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "ok": self.ok, "reward": self.reward, "problems": self.problems, "tests": self.table}


def check_run(kind: str, result: RunResult, manifest: dict[str, list[str]]) -> CheckReport:
    """Base: reward 0, every fail_to_pass fails for business reasons, the rest pass.
    Oracle: reward 1, everything passes."""
    report = CheckReport(kind=kind, ok=True, reward=result.reward,
                         table={tid: st.outcome for tid, st in result.tests.items()})
    if result.timed_out:
        report.problems.append(f"{kind}: run timed out")
    if result.reward is None:
        report.problems.append(f"{kind}: reward.txt missing or invalid (test.sh crashed?)")
    expected_reward = 0 if kind == "base" else 1
    if result.reward is not None and result.reward != expected_reward:
        report.problems.append(f"{kind}: reward={result.reward}, expected {expected_reward}")

    def status(tid: str) -> TestStatus:
        return result.tests.get(tid) or TestStatus(tid, "missing", "not in results")

    for tid in manifest["fail_to_pass"]:
        st = status(tid)
        if kind == "base":
            if st.outcome == "passed":
                report.problems.append(f"base: fail_to_pass test passed on original code: {tid}")
            elif st.outcome in ("skipped", "missing"):
                report.problems.append(f"base: fail_to_pass test was {st.outcome}: {tid}")
            elif st.outcome == "error" and _ENV_FAILURE.search(st.message or ""):
                report.problems.append(f"base: fail_to_pass test errored for environment/syntax reasons: {tid}: "
                                       f"{st.message[:MSG_LIMIT]}")
        elif st.outcome != "passed":
            report.problems.append(f"oracle: fail_to_pass test {st.outcome}: {tid}: {st.message[:MSG_LIMIT]}")
    for cat in ("pass_to_pass", "anti_cheat"):
        for tid in manifest[cat]:
            st = status(tid)
            if st.outcome != "passed":
                report.problems.append(f"{kind}: {cat} test {st.outcome}: {tid}: {st.message[:MSG_LIMIT]}")
    report.ok = not report.problems
    return report
