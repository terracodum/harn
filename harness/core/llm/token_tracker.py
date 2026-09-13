"""Accounting of every LLM call for evidence/llm_usage.json (Stage 3.4)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class LLMCall:
    purpose: str
    model: str
    duration_sec: float
    input_tokens: int | None        # None when the provider does not report usage (PROTOCOL.md)
    output_tokens: int | None
    ok: bool = True
    error: str | None = None
    finish_reason: str | None = None
    mode: str | None = None      # response_format mode actually used (json_schema | json_object | text)


@dataclass
class TokenTracker:
    calls: list[LLMCall] = field(default_factory=list)

    def record(self, call: LLMCall) -> None:
        self.calls.append(call)

    def summary(self, start: int = 0) -> dict:
        """Summary of the calls recorded from index `start` (a case's share of the run)."""
        calls = self.calls[start:]
        return {
            "total_calls": len(calls),
            "failed_calls": sum(1 for c in calls if not c.ok),
            "total_input_tokens": sum(c.input_tokens or 0 for c in calls),
            "total_output_tokens": sum(c.output_tokens or 0 for c in calls),
            "tokens_unknown_calls": sum(1 for c in calls if c.input_tokens is None or c.output_tokens is None),
            "total_duration_sec": round(sum(c.duration_sec for c in calls), 3),
            "models": sorted({c.model for c in calls}),
            "calls": [asdict(c) for c in calls],
        }

    def write(self, path: Path, start: int = 0) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(start), indent=2, ensure_ascii=False), encoding="utf-8")
