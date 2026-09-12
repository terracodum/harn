"""Stage 6: task.toml, evidence/summary.json, result.json."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.core.config import CaseConfig
from harness.providers.base import StackProfile


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        return "[\n" + "".join(f"    {_toml_value(v)},\n" for v in value) + "]"
    raise TypeError(f"unsupported TOML value: {type(value).__name__}")


def toml_dumps(data: dict[str, Any]) -> str:
    """Minimal TOML writer: scalars, arrays of scalars, one level of tables."""
    lines: list[str] = []
    tables: list[tuple[str, dict]] = []
    for key, value in data.items():
        if isinstance(value, dict):
            tables.append((key, value))
        else:
            lines.append(f"{key} = {_toml_value(value)}")
    for name, table in tables:
        lines.append("")
        lines.append(f"[{name}]")
        for key, value in table.items():
            lines.append(f"{key} = {_toml_value(value)}")
    return "\n".join(lines) + "\n"


def write_task_toml(task_dir: Path, config: CaseConfig, profile: StackProfile,
                    manifest: dict[str, list[str]], snapshot_sha256: str) -> None:
    doc: dict[str, Any] = {
        "schema_version": "1.1",
        "case_id": config.case_id,
        "language": profile.language,
        "authors": [config.author],
        "seed": config.seed,
        "input_snapshot_sha256": snapshot_sha256,
        "fail_to_pass": manifest["fail_to_pass"],
        "pass_to_pass": manifest["pass_to_pass"],
        "anti_cheat": manifest["anti_cheat"],
        "limits": {
            "build_timeout_sec": config.limits.build_timeout_sec,
            "run_timeout_sec": config.limits.run_timeout_sec,
        },
        "environment": {
            "dockerfile": "environment/Dockerfile",
            "repo": "environment/repo",
            "workdir": "/app/repo",
            "tests_mount": "/tests",
            "solution_mount": "/solution",
            "logs_mount": "/logs",
            "test_command": "sh /tests/test.sh",
            "reward_file": "/logs/verifier/reward.txt",
            "network": "none",
        },
        "solution": {"script": "solution/solve.sh", "apply_command": "sh /solution/solve.sh"},
    }
    (task_dir / "task.toml").write_text(toml_dumps(doc), encoding="utf-8", newline="\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def write_result(output_dir: Path, *, status: str, error: str | None, limitations: list[str],
                 attempts: int, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    result = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task_dir": "task",
        "evidence_dir": "evidence",
        "attempts": attempts,
        "error": error,
        "limitations": limitations,
        **(extra or {}),
    }
    write_json(output_dir / "result.json", result)
    return result
