"""LLM client abstraction. The pipeline only ever calls `complete_json`."""
from __future__ import annotations

import abc
import json
import re
from typing import Any

from harness.core.llm.token_tracker import TokenTracker


class LLMError(RuntimeError):
    pass


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply (tolerates fences/prose)."""
    candidates = [text.strip()]
    candidates += [m.strip() for m in _FENCE.findall(text)]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for cand in candidates:
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    raise LLMError("no JSON object found in model reply")


def check_required(data: dict[str, Any], schema: dict[str, Any]) -> None:
    missing = [k for k in schema.get("required", []) if k not in data]
    if missing:
        raise LLMError(f"model reply is missing required keys: {', '.join(missing)}")


class BaseLLMClient(abc.ABC):
    def __init__(self, tracker: TokenTracker | None = None) -> None:
        self.tracker = tracker or TokenTracker()

    @abc.abstractmethod
    def complete_json(self, *, purpose: str, system: str, user: str,
                      schema: dict[str, Any], max_tokens: int = 32_000) -> dict[str, Any]:
        """Return a JSON object shaped like `schema`.

        `purpose` is a short tag (e.g. "localize", "synthesize", "heal") used
        for usage accounting and by mock clients.
        """
