"""Stage 6: task.toml, evidence/summary.json, result.json (formats from PROTOCOL.md)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.core.config import CaseConfig


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):  # inline table
        return "{ " + ", ".join(f"{k} = {_toml_value(v)}" for k, v in value.items()) + " }"
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        if all(isinstance(v, dict) for v in value):
            return "[" + ", ".join(_toml_value(v) for v in value) + "]"
        return "[\n" + "".join(f"    {_toml_value(v)},\n" for v in value) + "]"
    raise TypeError(f"unsupported TOML value: {type(value).__name__}")


def toml_dumps(data: dict[str, Any]) -> str:
    """Minimal TOML writer: scalars, arrays of scalars, arrays of inline tables, one level of tables."""
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


def write_task_toml(task_dir: Path, config: CaseConfig, manifest: dict[str, list[str]], *,
                    description: str, bank_domain: str, name: str | None = None) -> None:
    """PROTOCOL.md section 4: schema_version, [task], [metadata] (with the three test lists),
    [agent], [verifier], [environment]. `name` defaults to the input case_id."""
    doc: dict[str, Any] = {
        "schema_version": "1.1",
        "task": {
            "name": name or config.case_id,
            "description": description,
            "authors": [config.author.to_dict()],
        },
        "metadata": {
            "task_type": "agentic",
            "bank_domain": bank_domain,
            "language": config.language,
            "build_tool": "docker",
            "difficulty": config.difficulty,
            "source": config.source,
            "team": config.team,
            "fail_to_pass": list(manifest["fail_to_pass"]),
            "pass_to_pass": list(manifest["pass_to_pass"]),
            "anti_cheat": list(manifest["anti_cheat"]),
        },
        "agent": {"timeout_sec": config.limits.agent_timeout_sec},
        "verifier": {"timeout_sec": config.limits.verifier_timeout_sec},
        "environment": {
            "allow_internet": False,
            "build_timeout_sec": config.limits.build_timeout_sec,
            "cpus": config.limits.cpus,
            "memory_mb": config.limits.memory_mb,
            "storage_mb": config.limits.storage_mb,
        },
    }
    (task_dir / "task.toml").write_text(toml_dumps(doc), encoding="utf-8", newline="\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def write_result(output_dir: Path, *, config: CaseConfig, status: str, error: str | None,
                 limitations: list[str], attempts: int, snapshot_sha256: str,
                 case_id: str | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """result.json of ONE case (PROTOCOL.md section 2). `status` is ready or failed; a case that was
    not verified is failed, with the reason in `limitations`. Extra harness fields (error, attempts,
    failed_stage, llm, case_status) are additive."""
    if status not in ("ready", "failed"):
        raise ValueError(f"result status must be ready or failed, got {status!r}")
    task_ok = (output_dir / "task" / "task.toml").exists()
    evidence_ok = (output_dir / "evidence").is_dir()
    lims = list(limitations)
    if error and error not in lims:
        lims.append(error)
    result = {
        "protocol_version": config.protocol_version,
        "case_id": case_id or config.case_id,
        "status": status,
        "task_path": "task" if task_ok else None,
        "evidence_path": "evidence" if evidence_ok else None,
        "limitations": lims,
        "input_snapshot_sha256": snapshot_sha256 or None,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "attempts": attempts,
        "error": error,
        **(extra or {}),
    }
    write_json(output_dir / "result.json", result)
    return result


def write_run_result(output_dir: Path, *, config: CaseConfig, status: str, error: str | None,
                     limitations: list[str], snapshot_sha256: str, cases: list[dict[str, Any]],
                     extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """result.json of the whole run: the protocol fields (task_path is null - the cases live under
    cases/<R>/ with a result.json each) plus the list of cases with their status (ready | case_failed).
    The run is ready only when every case is ready; failed cases are always listed."""
    if status not in ("ready", "failed"):
        raise ValueError(f"result status must be ready or failed, got {status!r}")
    lims = list(limitations)
    for c in cases:
        if c["status"] != "ready":
            line = f"case {c['requirement_id']} ({c['title']}) is case_failed: {c.get('error') or 'not verified'}"
            if line not in lims:
                lims.append(line)
    if error and error not in lims:
        lims.append(error)
    result = {
        "protocol_version": config.protocol_version,
        "case_id": config.case_id,
        "status": status,
        "task_path": None,
        "evidence_path": "evidence" if (output_dir / "evidence").is_dir() else None,
        "cases_path": "cases" if cases else None,
        "limitations": lims,
        "input_snapshot_sha256": snapshot_sha256 or None,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cases": cases,
        "cases_ready": sum(1 for c in cases if c["status"] == "ready"),
        "cases_failed": sum(1 for c in cases if c["status"] != "ready"),
        "attempts": sum(int(c.get("attempts") or 0) for c in cases),
        "error": error,
        **(extra or {}),
    }
    write_json(output_dir / "result.json", result)
    return result
