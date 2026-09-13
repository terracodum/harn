"""Client for any OpenAI-compatible chat endpoint (Ollama, vLLM, LM Studio,
GigaChat via an OpenAI-compatible proxy, ...).

JSON is requested in the most constrained mode the server supports:
  json_schema -> json_object -> plain text with a JSON instruction.
A mode is skipped only when the server answers 400 and names `response_format` (or the
schema) as the reason; the switch is logged and recorded in llm_usage.json. Any other
error is raised as `LLMError` - the client never re-sends a request under a weaker
format to paper over an unrelated failure.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from harness.core.config import LLMSettings
from harness.core.llm.base_client import BaseLLMClient, LLMError, check_required, extract_json_object
from harness.core.llm.token_tracker import LLMCall, TokenTracker

_MODES = ("json_schema", "json_object", "text")
_UNSUPPORTED_FORMAT = re.compile(r"response_format|json_schema|json_object|structured output|structured_output|"
                                 r"guided_json|schema", re.IGNORECASE)
log = logging.getLogger(__name__)


def _tokens(usage: Any, field: str) -> int | None:
    """Token count as reported by the server, None when it is not reported (never a made-up 0)."""
    value = getattr(usage, field, None) if usage is not None else None
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


class OpenAICompatClient(BaseLLMClient):
    def __init__(self, settings: LLMSettings, tracker: TokenTracker | None = None) -> None:
        super().__init__(tracker)
        try:
            import openai
        except ImportError as exc:  # pragma: no cover
            raise LLMError("pip install openai (or run with --mock-llm)") from exc
        self._openai = openai
        self.settings = settings
        self._client = openai.OpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout=float(settings.timeout_sec),
            max_retries=2,
            http_client=openai.DefaultHttpxClient(trust_env=settings.trust_env),
        )
        self._unsupported: set[str] = set()

    # ------------------------------------------------------------------ helpers
    def _response_format(self, mode: str, purpose: str, schema: dict[str, Any]) -> dict[str, Any] | None:
        if mode == "json_schema":
            return {"type": "json_schema",
                    "json_schema": {"name": purpose, "schema": schema, "strict": False}}
        if mode == "json_object":
            return {"type": "json_object"}
        return None

    def _call(self, messages: list[dict[str, str]], mode: str, purpose: str,
              schema: dict[str, Any], max_tokens: int) -> tuple[str, Any]:
        kwargs: dict[str, Any] = dict(
            model=self.settings.model,
            messages=messages,
            temperature=self.settings.temperature,
            max_tokens=max_tokens,
        )
        fmt = self._response_format(mode, purpose, schema)
        if fmt:
            kwargs["response_format"] = fmt
        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        return (choice.message.content or ""), resp

    # -------------------------------------------------------------------- API
    def complete_json(self, *, purpose: str, system: str, user: str,
                      schema: dict[str, Any], max_tokens: int = 32_000) -> dict[str, Any]:
        schema_hint = ("\n\nReply with a single JSON object only, no prose, matching this JSON schema:\n"
                       + json.dumps(schema, ensure_ascii=False))
        messages = [
            {"role": "system", "content": system + schema_hint},
            {"role": "user", "content": user},
        ]
        last_error: Exception | None = None
        for mode in _MODES:
            if mode in self._unsupported:
                continue
            for attempt in range(2):
                started = time.monotonic()
                try:
                    text, resp = self._call(messages, mode, purpose, schema, max_tokens)
                except self._openai.BadRequestError as exc:
                    self.tracker.record(LLMCall(purpose, self.settings.model, time.monotonic() - started,
                                                None, None, ok=False, error=f"{type(exc).__name__}: {exc}", mode=mode))
                    if mode == "text" or not _UNSUPPORTED_FORMAT.search(str(exc)):
                        # an unrelated 400 (context too long, bad model id, ...): never mask it by
                        # re-sending the same request under a weaker response_format
                        raise LLMError(f"LLM call '{purpose}' rejected by the server (HTTP 400): {exc}") from exc
                    log.warning("server rejected response_format=%s for '%s' (%s); switching to the next mode",
                                mode, purpose, str(exc)[:200])
                    self._unsupported.add(mode)
                    last_error = exc
                    break
                except self._openai.APIError as exc:
                    self.tracker.record(LLMCall(purpose, self.settings.model, time.monotonic() - started,
                                                None, None, ok=False, error=f"{type(exc).__name__}: {exc}"))
                    raise LLMError(f"LLM call '{purpose}' failed: {exc}") from exc

                usage = getattr(resp, "usage", None)
                call = LLMCall(
                    purpose=purpose,
                    model=getattr(resp, "model", None) or self.settings.model,
                    duration_sec=round(time.monotonic() - started, 3),
                    input_tokens=_tokens(usage, "prompt_tokens"),
                    output_tokens=_tokens(usage, "completion_tokens"),
                    finish_reason=resp.choices[0].finish_reason,
                    mode=mode,
                )
                self.tracker.record(call)
                if call.finish_reason == "length":
                    call.ok = False
                    call.error = "max_tokens"
                    raise LLMError(f"LLM output truncated (max_tokens={max_tokens}) for '{purpose}'")
                try:
                    data = extract_json_object(text)
                    check_required(data, schema)
                    return data
                except LLMError as exc:
                    call.ok = False
                    call.error = str(exc)
                    last_error = exc
                    if attempt == 0:
                        messages = messages + [
                            {"role": "assistant", "content": text[:20_000]},
                            {"role": "user", "content": f"That was not valid: {exc}. "
                                                        "Return ONLY the JSON object, complete and valid."},
                        ]
            else:
                continue
        raise LLMError(f"LLM call '{purpose}' produced no usable JSON: {last_error}")
