"""Validation of the input JSON and limits (Stage 1.1)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Limits:
    build_timeout_sec: int = 1800
    run_timeout_sec: int = 900
    max_retries: int = 2          # self-healing iterations
    max_context_files: int = 8    # top-N implementation files passed to the LLM
    max_context_chars: int = 80_000

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "Limits":
        raw = raw or {}
        kwargs: dict[str, Any] = {}
        for name in cls.__dataclass_fields__:
            if name in raw:
                value = raw[name]
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ConfigError(f"limits.{name} must be a non-negative integer")
                kwargs[name] = value
        return cls(**kwargs)


@dataclass(frozen=True)
class LLMSettings:
    """OpenAI-compatible chat + embeddings endpoint (Ollama, vLLM, GigaChat proxy, ...).

    Resolution order: CLI flag > input JSON "llm" block > environment variables.
    """
    model: str = "gpt-oss:120b"
    base_url: str | None = None            # e.g. http://localhost:11434/v1
    api_key_env: str = "LLM_API_KEY"       # name of the env var holding the key
    embedding_model: str | None = None     # e.g. nomic-embed-text, bge-m3; None = FTS only
    embedding_base_url: str | None = None  # defaults to base_url
    temperature: float = 0.0
    timeout_sec: int = 1800
    # Honour HTTP(S)_PROXY / Windows system proxy for LLM traffic. Off by default: a system
    # proxy silently swallows requests to a local Ollama/vLLM and answers 503.
    trust_env: bool = False

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None, *, overrides: dict[str, Any] | None = None) -> "LLMSettings":
        raw = dict(raw or {})
        env_defaults = {
            "model": os.environ.get("LLM_MODEL"),
            "base_url": os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL"),
            "embedding_model": os.environ.get("LLM_EMBED_MODEL"),
            "embedding_base_url": os.environ.get("LLM_EMBED_BASE_URL"),
            "trust_env": os.environ.get("LLM_TRUST_ENV", "").lower() in ("1", "true", "yes") or None,
        }
        merged: dict[str, Any] = {k: v for k, v in env_defaults.items() if v}
        merged.update({k: v for k, v in raw.items() if v is not None})
        merged.update({k: v for k, v in (overrides or {}).items() if v is not None})
        unknown = set(merged) - set(cls.__dataclass_fields__)
        if unknown:
            raise ConfigError(f"unknown llm settings: {', '.join(sorted(unknown))}")
        return cls(**merged)

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env) or os.environ.get("OPENAI_API_KEY") or "not-needed"


@dataclass(frozen=True)
class CaseConfig:
    case_id: str
    brief: str
    repository: Path
    output_dir: Path
    author: str
    seed: int
    limits: Limits = field(default_factory=Limits)
    llm: LLMSettings = field(default_factory=LLMSettings)
    language: str = "python"
    untrusted_dirs: tuple[str, ...] = ()
    source: dict[str, Any] = field(default_factory=dict)

    REQUIRED = ("case_id", "brief", "repository", "limits", "author", "seed")

    @classmethod
    def load(cls, path: str | Path, *, output_dir: str | Path | None = None,
             llm_overrides: dict[str, Any] | None = None) -> "CaseConfig":
        path = Path(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"cannot read input JSON {path}: {exc}") from exc
        # a CLI-supplied output_dir is relative to the current directory, JSON paths to the JSON file
        cli_out = Path(output_dir).resolve() if output_dir else None
        return cls.from_dict(raw, base_dir=path.parent.resolve(), output_dir=cli_out,
                             llm_overrides=llm_overrides)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, base_dir: Path = Path("."),
                  output_dir: str | Path | None = None,
                  llm_overrides: dict[str, Any] | None = None) -> "CaseConfig":
        if not isinstance(raw, dict):
            raise ConfigError("input JSON must be an object")
        missing = [k for k in cls.REQUIRED if k not in raw]
        if missing:
            raise ConfigError(f"input JSON is missing required keys: {', '.join(missing)}")

        case_id = raw["case_id"]
        if not isinstance(case_id, str) or not case_id.strip():
            raise ConfigError("case_id must be a non-empty string")
        if any(ch in case_id for ch in "/\\ \t\n:"):
            raise ConfigError("case_id must not contain path separators, colons or whitespace")
        brief = raw["brief"]
        if not isinstance(brief, str) or len(brief.strip()) < 10:
            raise ConfigError("brief must be a meaningful non-empty string")
        author = raw["author"]
        if not isinstance(author, str) or not author.strip():
            raise ConfigError("author must be a non-empty string")
        seed = raw["seed"]
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ConfigError("seed must be an integer")

        repo = Path(raw["repository"])
        if not repo.is_absolute():
            repo = (base_dir / repo).resolve()
        if not repo.is_dir():
            raise ConfigError(f"repository directory does not exist: {repo}")

        out = output_dir or raw.get("output_dir")
        if not out:
            raise ConfigError("output_dir must be given in JSON or via --output-dir")
        out = Path(out)
        if not out.is_absolute():
            out = (base_dir / out).resolve()
        if out.exists() and any(out.iterdir()):
            raise ConfigError(f"output_dir already exists and is not empty: {out}")
        try:
            out.resolve().relative_to(repo.resolve())
        except ValueError:
            pass
        else:
            raise ConfigError("output_dir must not be inside the source repository")

        untrusted = tuple(raw.get("untrusted_dirs") or ())
        if not all(isinstance(u, str) for u in untrusted):
            raise ConfigError("untrusted_dirs must be a list of strings")

        return cls(
            case_id=case_id.strip(),
            brief=brief.strip(),
            repository=repo,
            output_dir=out,
            author=author.strip(),
            seed=seed,
            limits=Limits.from_dict(raw.get("limits")),
            llm=LLMSettings.from_dict(raw.get("llm"), overrides=llm_overrides),
            language=str(raw.get("language", "python")),
            untrusted_dirs=untrusted,
            source=raw,
        )
