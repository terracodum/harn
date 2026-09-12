"""Canned-response client for offline runs and tests.

Responses file: {"<purpose>": <json object> | [<json object>, ...]}.
A list is consumed in order (useful for modelling self-healing rounds);
the last element is repeated once exhausted.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harness.core.llm.base_client import BaseLLMClient, LLMError, check_required
from harness.core.llm.token_tracker import LLMCall, TokenTracker


class MockLLMClient(BaseLLMClient):
    def __init__(self, responses: dict[str, Any], tracker: TokenTracker | None = None) -> None:
        super().__init__(tracker)
        self._responses = responses
        self._cursor: dict[str, int] = {}
        self.prompts: list[dict[str, str]] = []  # captured for tests

    @classmethod
    def from_file(cls, path: str | Path, tracker: TokenTracker | None = None) -> "MockLLMClient":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), tracker)

    def complete_json(self, *, purpose: str, system: str, user: str,
                      schema: dict[str, Any], max_tokens: int = 32_000) -> dict[str, Any]:
        self.prompts.append({"purpose": purpose, "system": system, "user": user})
        if purpose not in self._responses:
            raise LLMError(f"mock has no response for purpose '{purpose}'")
        value = self._responses[purpose]
        if isinstance(value, list):
            idx = min(self._cursor.get(purpose, 0), len(value) - 1)
            self._cursor[purpose] = idx + 1
            value = value[idx]
        data = json.loads(json.dumps(value))
        check_required(data, schema)
        self.tracker.record(LLMCall(purpose, "mock", 0.0, len(user) // 4,
                                    len(json.dumps(value)) // 4, finish_reason="stop"))
        return data
