"""Stage 5: build image, Base run, Oracle run, isolated category runs, checks."""
from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from harness.core.config import Limits
from harness.providers.base import ITestRunner, RunResult
from harness.providers.python.test_runner import CheckReport, check_run
from harness.providers.python.verifier_template import CATEGORIES
from harness.validation.docker_runner import DockerRunner, Mount, ProcResult

log = logging.getLogger(__name__)

_ENV_REASON = re.compile(r"async def functions are not natively supported|SyntaxError|IndentationError|"
                         r"ModuleNotFoundError|ImportError|fixture '.*' not found|No module named", re.IGNORECASE)


def solution_change_justified(report: "VerificationReport") -> bool:
    """solve.sh may only be rewritten by the healing round when the ORACLE side complained for a
    business reason: a fail_to_pass test fails after the solution with a real assertion (the fix is
    incomplete) or solve.sh itself crashed. Base-side failures and environment errors (missing
    plugin, syntax/import errors in a test) are test defects, never a reason to touch the solution."""
    for p in report.problems:
        if not p.startswith("oracle:"):
            continue
        if "solve.sh exited" in p:
            return True
        if "fail_to_pass test" in p and not _ENV_REASON.search(p):
            return True
    return False


def auto_recategorize(report: "VerificationReport", manifest: dict[str, list[str]]) -> list[str]:
    """Deterministic fix that needs no LLM: a fail_to_pass test that passes both on the original
    code and after the solution is by definition pass_to_pass. Moves such tests and returns the
    moved ids. Only applies when EVERY reported problem is of that single kind."""
    if report.base is None or report.oracle is None:
        return []
    leaky = [t for t in manifest["fail_to_pass"]
             if report.base.table.get(t) == "passed" and report.oracle.table.get(t) == "passed"]
    if not leaky:
        return []
    explained = {f"base: fail_to_pass test passed on original code: {t}" for t in leaky}
    other = [p for p in report.problems if p not in explained and not p.startswith("base: reward=")]
    if other or len(leaky) == len(manifest["fail_to_pass"]):
        return []          # something else is wrong, or nothing would be left in fail_to_pass
    manifest["fail_to_pass"] = [t for t in manifest["fail_to_pass"] if t not in leaky]
    manifest["pass_to_pass"] = manifest["pass_to_pass"] + leaky
    return leaky


_ORACLE_CMD = ("sh /solution/solve.sh >/logs/solve.log 2>&1; echo $? >/logs/solve_exit.txt; "
               "sh /tests/test.sh {category}")
_BASE_CMD = "sh /tests/test.sh {category}"


def _tail(path: Path, limit: int = 6000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-limit:]


@dataclass
class VerificationReport:
    ok: bool = False
    build_ok: bool = False
    problems: list[str] = field(default_factory=list)
    base: CheckReport | None = None
    oracle: CheckReport | None = None
    isolated: dict[str, dict[str, dict]] = field(default_factory=dict)
    runs: list[dict] = field(default_factory=list)      # entries for evidence/summary.json
    logs: dict[str, str] = field(default_factory=dict)  # excerpts for the healing prompt

    def to_dict(self) -> dict:
        return {
            "ok": self.ok, "build_ok": self.build_ok, "problems": self.problems,
            "base": self.base.to_dict() if self.base else None,
            "oracle": self.oracle.to_dict() if self.oracle else None,
            "isolated": self.isolated, "runs": self.runs,
        }


class ConvergenceVerifier:
    def __init__(self, docker: DockerRunner, runner: ITestRunner, *, task_dir: Path, evidence_dir: Path,
                 limits: Limits, image_tag: str, isolated_runs: bool = True) -> None:
        self.docker = docker
        self.runner = runner
        self.task_dir = task_dir
        self.evidence_dir = evidence_dir
        self.limits = limits
        self.image_tag = image_tag
        self.isolated_runs = isolated_runs

    # ----------------------------------------------------------------- pieces
    def build_image(self) -> ProcResult:
        return self.docker.build(self.task_dir / "environment", self.image_tag,
                                 log_path=self.evidence_dir / "build.log",
                                 timeout=self.limits.build_timeout_sec)

    def run_case(self, name: str, *, with_solution: bool, category: str, logs_dir: Path,
                 manifest: dict[str, list[str]]) -> tuple[ProcResult, RunResult]:
        if logs_dir.exists():
            shutil.rmtree(logs_dir)
        (logs_dir / "verifier").mkdir(parents=True)
        mounts = [Mount(self.task_dir / "tests", "/tests", True), Mount(logs_dir, "/logs", False)]
        if with_solution:
            mounts.append(Mount(self.task_dir / "solution", "/solution", True))
        cmd = (_ORACLE_CMD if with_solution else _BASE_CMD).format(category=category)
        proc = self.docker.run(self.image_tag, mounts=mounts, command=cmd,
                               log_path=logs_dir / "container.log", timeout=self.limits.run_timeout_sec,
                               cpus=self.limits.cpus, memory_mb=self.limits.memory_mb)
        result = self.runner.parse_results(logs_dir, manifest, category)
        result.exit_code, result.duration_sec, result.timed_out = proc.exit_code, proc.duration_sec, proc.timed_out
        if with_solution:
            exit_file = logs_dir / "solve_exit.txt"
            code = exit_file.read_text(encoding="utf-8").strip() if exit_file.exists() else "?"
            if code != "0":
                result.notes.append(f"solve.sh exited with {code}")
        return proc, result

    # ------------------------------------------------------------------ verify
    def verify(self, manifest: dict[str, list[str]], *, need_build: bool) -> VerificationReport:
        report = VerificationReport()

        if need_build:
            build = self.build_image()
            report.runs.append({"name": "build", **build.to_dict(), "status": "ok" if build.exit_code == 0 else "failed"})
            if build.exit_code != 0:
                report.problems.append("docker build failed" + (" (timeout)" if build.timed_out else ""))
                report.logs["build.log"] = _tail(self.evidence_dir / "build.log")
                return report
        report.build_ok = True

        for kind, with_solution in (("base", False), ("oracle", True)):
            logs_dir = self.evidence_dir / kind
            proc, result = self.run_case(kind, with_solution=with_solution, category="all",
                                         logs_dir=logs_dir, manifest=manifest)
            check = check_run(kind, result, manifest)
            for note in result.notes:
                check.problems.append(f"{kind}: {note}")
                check.ok = False
            setattr(report, kind, check)
            report.runs.append({"name": kind, **proc.to_dict(), "reward": result.reward,
                                "status": "ok" if check.ok else "failed"})
            report.problems.extend(check.problems)
            report.logs[f"{kind}/pytest.log"] = _tail(logs_dir / "verifier" / "pytest.log")
            if with_solution:
                report.logs["oracle/solve.log"] = _tail(logs_dir / "solve.log", 3000)
            if not check.ok and kind == "base":
                report.logs["base/container.log"] = _tail(logs_dir / "container.log", 2000)

        if report.problems:
            return report

        if self.isolated_runs:
            for kind, with_solution in (("base", False), ("oracle", True)):
                report.isolated[kind] = {}
                for cat in CATEGORIES:
                    if not manifest[cat]:
                        continue
                    logs_dir = self.evidence_dir / "isolated" / kind / cat
                    proc, result = self.run_case(f"{kind}/{cat}", with_solution=with_solution, category=cat,
                                                 logs_dir=logs_dir, manifest=manifest)
                    expected = 0 if (kind == "base" and cat == "fail_to_pass") else 1
                    ok = result.reward == expected and not result.timed_out
                    report.isolated[kind][cat] = {"reward": result.reward, "expected": expected, "ok": ok,
                                                  "duration_sec": proc.duration_sec}
                    report.runs.append({"name": f"isolated/{kind}/{cat}", **proc.to_dict(),
                                        "reward": result.reward, "status": "ok" if ok else "failed"})
                    if not ok:
                        report.problems.append(f"isolated {kind}/{cat}: reward={result.reward}, expected {expected}")
        report.ok = not report.problems
        return report
