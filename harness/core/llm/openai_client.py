"""Client for any OpenAI-compatible chat endpoint (Ollama, vLLM, LM Studio,
GigaChat via an OpenAI-compatible proxy, ...).

JSON is requested in the most constrained mode the server supports:
  json_schema -> json_object -> plain text with a JSON instruction.
Unsupported modes are detected on the first 400 and skipped afterwards.
"""
from __future__ import annotations

import json
import time
from typing import Any

from harness.core.config import LLMSettings
from harness.core.llm.base_client import BaseLLMClient, LLMError, check_required, extract_json_object
from harness.core.llm.token_tracker import LLMCall, TokenTracker

_MODES = ("json_schema", "json_object", "text")


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
                    # server does not support this response_format -> try the next mode
                    self._unsupported.add(mode)
                    last_error = exc
                    self.tracker.record(LLMCall(purpose, self.settings.model, time.monotonic() - started,
                                                0, 0, ok=False, error=f"{mode}: {exc}"))
                    break
                except self._openai.APIError as exc:
                    self.tracker.record(LLMCall(purpose, self.settings.model, time.monotonic() - started,
                                                0, 0, ok=False, error=f"{type(exc).__name__}: {exc}"))
                    raise LLMError(f"LLM call '{purpose}' failed: {exc}") from exc

                usage = getattr(resp, "usage", None)
                call = LLMCall(
                    purpose=purpose,
                    model=getattr(resp, "model", None) or self.settings.model,
                    duration_sec=round(time.monotonic() - started, 3),
                    input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                    output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                    finish_reason=resp.choices[0].finish_reason,
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
