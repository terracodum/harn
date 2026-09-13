"""End-to-end without Docker: mock LLM, --skip-docker, then the generated verify.py
is executed on the host against the demo repo (Base) and after solve.sh (Oracle)."""
import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from harness.core.config import CaseConfig
from harness.core.llm.mock_client import MockLLMClient
from harness.core.pipeline import Pipeline
from harness.core.snapshot import snapshot_sha256
from harness.providers.python.test_runner import PytestJUnitRunner, check_run


@pytest.fixture
def generated(demo_input, mock_responses, tmp_path):
    cfg = CaseConfig.load(demo_input)
    before = snapshot_sha256(cfg.repository)
    result = Pipeline(cfg, MockLLMClient(mock_responses), search_backend="keyword", skip_docker=True).run()
    assert snapshot_sha256(cfg.repository) == before, "source repo must stay untouched"
    return cfg, result


def test_pipeline_packages_task(generated):
    cfg, result = generated
    assert result["status"] == "failed" and result["error"] is None, result       # packaged but not verified
    assert result["task_path"] is None and result["evidence_path"] == "evidence" and result["cases_path"] == "cases"
    assert result["protocol_version"] == "1.0" and result["case_id"] == cfg.case_id
    assert [c["requirement_id"] for c in result["cases"]] == ["R1"]
    case = result["cases"][0]
    assert case["status"] == "case_failed" and case["task_path"] == "cases/R1/task"
    assert any("NOT verified" in lim for lim in case["limitations"])
    out = cfg.output_dir / "cases" / "R1"
    task = out / "task"
    for rel in ("task.toml", "instruction.md", "solution/solve.sh", "tests/test.sh", "tests/verify.py",
                "tests/manifest.json", "tests/pytest.ini", "tests/test_netting_refunds.py",
                "environment/Dockerfile", "environment/repo/ledger/netting.py"):
        assert (task / rel).exists(), rel
    assert not (task / "environment/repo/__pycache__").exists()
    assert not (out / ".work").exists()
    toml = tomllib.loads((task / "task.toml").read_text(encoding="utf-8"))
    assert toml["schema_version"] == "1.1"
    assert result["input_snapshot_sha256"] == snapshot_sha256(cfg.repository)
    assert toml["task"]["name"] == f"{cfg.case_id}-r1" and toml["task"]["authors"][0]["email"] == cfg.author.email
    md = toml["metadata"]
    assert md["task_type"] == "agentic" and md["bank_domain"] and md["language"] == "ru"
    assert len(md["fail_to_pass"]) == 3 and len(md["anti_cheat"]) == 2
    assert set(md["fail_to_pass"]).isdisjoint(md["pass_to_pass"])
    assert toml["environment"]["allow_internet"] is False
    usage = json.loads((cfg.output_dir / "evidence/llm_usage.json").read_text())
    assert usage["total_calls"] == 2
    assert {c["purpose"] for c in usage["calls"]} == {"analyze_brief", "synthesize"}
    assert json.loads((out / "evidence/llm_usage.json").read_text())["total_calls"] == 1
    assert (cfg.output_dir / "evidence/profile.json").exists() and (out / "evidence/summary.json").exists()
    spec = json.loads((cfg.output_dir / "evidence/brief_spec.json").read_text(encoding="utf-8"))
    assert [r["id"] for r in spec["requirements"]] == ["R1", "R2"]
    synth = json.loads((out / "evidence/synthesis.json").read_text(encoding="utf-8"))
    assert {c["requirement_id"] for c in synth["coverage"]} == {"R1", "R2"}
    case_result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert case_result["case_id"] == f"{cfg.case_id}-r1" and case_result["task_path"] == "task"
    dockerfile = (task / "environment/Dockerfile").read_text()
    assert "pip install -e ." in dockerfile and "postgresql" not in dockerfile
    assert b"\r\n" not in (task / "tests/test.sh").read_bytes()


def _run_verify(task: Path, repo: Path, logs: Path, category: str = "all") -> dict:
    tests_dir = task / "tests"
    cmd = [sys.executable, str(tests_dir / "verify.py"), "--category", category,
           "--manifest", str(tests_dir / "manifest.json"), "--tests-dir", str(tests_dir),
           "--logs-dir", str(logs), "--cwd", str(repo), "--rootdir", str(task)]
    env = {"PYTHONPATH": str(repo), "PATH": __import__("os").environ["PATH"],
           "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads((logs / "results.json").read_text())


def test_verifier_base_and_oracle_on_host(generated, tmp_path):
    cfg, _ = generated
    task = cfg.output_dir / "cases" / "R1" / "task"
    manifest = json.loads((task / "tests/manifest.json").read_text())

    # Base run: original code
    repo = tmp_path / "base_repo"
    shutil.copytree(task / "environment/repo", repo)
    base_logs = tmp_path / "base"
    (base_logs / "verifier").mkdir(parents=True)
    res = _run_verify(task, repo, base_logs / "verifier")
    assert res["reward"] == 0
    run = PytestJUnitRunner().parse_results(base_logs, manifest, "all")
    report = check_run("base", run, manifest)
    assert report.ok, report.problems

    # Oracle run: apply solve.sh (needs a POSIX sh, e.g. Git Bash on Windows)
    sh = shutil.which("sh")
    if not sh:
        pytest.skip("no POSIX sh available to apply solve.sh")
    proc = subprocess.run([sh, str(task / "solution/solve.sh")], cwd=repo, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "total -= tx.amount" in (repo / "ledger/netting.py").read_text()
    oracle_logs = tmp_path / "oracle"
    (oracle_logs / "verifier").mkdir(parents=True)
    res = _run_verify(task, repo, oracle_logs / "verifier")
    assert res["reward"] == 1
    run = PytestJUnitRunner().parse_results(oracle_logs, manifest, "all")
    assert check_run("oracle", run, manifest).ok

    # Isolated category run on base: fail_to_pass alone must give reward 0
    repo2 = tmp_path / "base_repo2"
    shutil.copytree(task / "environment/repo", repo2)
    iso = tmp_path / "iso" / "verifier"
    iso.mkdir(parents=True)
    assert _run_verify(task, repo2, iso, "pass_to_pass")["reward"] == 1
    assert _run_verify(task, repo2, iso, "fail_to_pass")["reward"] == 0
