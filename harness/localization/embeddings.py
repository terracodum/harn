"""Dense embeddings through an OpenAI-compatible `/v1/embeddings` endpoint."""
from __future__ import annotations

from typing import Protocol

from harness.core.config import LLMSettings


class Embedder(Protocol):
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    def __init__(self, settings: LLMSettings, *, batch_size: int = 32, max_chars: int = 6000) -> None:
        import openai

        if not settings.embedding_model:
            raise ValueError("embedding_model is not configured")
        self.model = settings.embedding_model
        self.batch_size = batch_size
        self.max_chars = max_chars
        self._client = openai.OpenAI(
            base_url=settings.embedding_base_url or settings.base_url,
            api_key=settings.api_key,
            timeout=float(settings.timeout_sec),
            max_retries=2,
            http_client=openai.DefaultHttpxClient(trust_env=settings.trust_env),
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t[: self.max_chars] or " " for t in texts[i:i + self.batch_size]]
            resp = self._client.embeddings.create(model=self.model, input=batch)
            out.extend([d.embedding for d in sorted(resp.data, key=lambda d: d.index)])
        return out


def make_embedder(settings: LLMSettings) -> Embedder | None:
    return OpenAIEmbedder(settings) if settings.embedding_model else None
