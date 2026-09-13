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
    input_tokens: int
    output_tokens: int
    ok: bool = True
    error: str | None = None
    finish_reason: str | None = None
    mode: str | None = None      # response_format mode actually used (json_schema | json_object | text)


@dataclass
class TokenTracker:
    calls: list[LLMCall] = field(default_factory=list)

    def record(self, call: LLMCall) -> None:
        self.calls.append(call)

    def summary(self) -> dict:
        return {
            "total_calls": len(self.calls),
            "failed_calls": sum(1 for c in self.calls if not c.ok),
            "total_input_tokens": sum(c.input_tokens for c in self.calls),
            "total_output_tokens": sum(c.output_tokens for c in self.calls),
            "total_duration_sec": round(sum(c.duration_sec for c in self.calls), 3),
            "models": sorted({c.model for c in self.calls}),
            "calls": [asdict(c) for c in self.calls],
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), indent=2, ensure_ascii=False), encoding="utf-8")
