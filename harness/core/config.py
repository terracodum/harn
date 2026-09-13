"""Validation of the input JSON and limits (Stage 1.1).

Input format (protocol 1.0):

    {
      "protocol_version": "1.0",
      "repository": "../repo",            # relative to the JSON file
      "brief": "...",                     # task description (any language)
      "output_dir": "../runs/example",
      "case_id": "hackathon/settlement-001",
      "difficulty": "medium",
      "language": "ru",                   # language of instruction.md
      "limits": {"agent_timeout_sec": 1800, "verifier_timeout_sec": 300, "build_timeout_sec": 900,
                 "cpus": 2, "memory_mb": 4096, "storage_mb": 10240},
      "author": {"name": "...", "email": "..."},
      "seed": 4107,
      "source": "hackathon/settlement-001",
      "team": "team-example",
      "llm": {...}                        # optional, harness-specific (see LLMSettings)
    }
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_LANG_NAMES = {"ru": "Russian", "en": "English", "de": "German", "fr": "French", "es": "Spanish", "zh": "Chinese"}


def _load_brief(value: Any, base_dir: Path) -> str:
    """Accept the brief as text, as a list of paragraphs, or as a path to a .md/.txt file."""
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        value = "\n\n".join(v.strip() for v in value if v.strip())
    if not isinstance(value, str):
        raise ConfigError("brief must be a string, a list of strings, or a path to a text file")
    text = value.strip()
    if text and "\n" not in text and len(text) < 260 and text.lower().endswith((".md", ".txt", ".rst")):
        candidate = Path(text)
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8", errors="replace").strip()
    if len(text) < 10:
        raise ConfigError("brief must be a meaningful non-empty string (or a path to a brief file)")
    return text


def _int(raw: dict[str, Any], key: str, default: int, *, minimum: int = 0) -> int:
    value = raw.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ConfigError(f"limits.{key} must be an integer >= {minimum}")
    return value


def _bool(raw: dict[str, Any], key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"limits.{key} must be true or false")
    return value


@dataclass(frozen=True)
class Limits:
    # protocol fields
    agent_timeout_sec: int = 1800        # budget of the solver agent (recorded in task.toml)
    verifier_timeout_sec: int = 300      # one sandbox run (test.sh)
    build_timeout_sec: int = 900         # docker build
    cpus: int = 2
    memory_mb: int = 4096
    storage_mb: int = 10240
    # harness-specific extras (optional)
    max_retries: int = 3                 # self-healing iterations
    max_context_files: int = 10          # top-N implementation files passed to the LLM
    max_context_chars: int = 120_000
    # When healing is exhausted, drop the requirements that never converged (their solve step,
    # tests and paragraph of instruction.md) and ship the rest, recorded in result.json limitations.
    prune_unconverged: bool = True

    @property
    def run_timeout_sec(self) -> int:
        return self.verifier_timeout_sec

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "Limits":
        raw = dict(raw or {})
        if not isinstance(raw, dict):
            raise ConfigError("limits must be an object")
        if "run_timeout_sec" in raw and "verifier_timeout_sec" not in raw:   # legacy alias
            raw["verifier_timeout_sec"] = raw.pop("run_timeout_sec")
        defaults = cls()
        return cls(
            agent_timeout_sec=_int(raw, "agent_timeout_sec", defaults.agent_timeout_sec, minimum=1),
            verifier_timeout_sec=_int(raw, "verifier_timeout_sec", defaults.verifier_timeout_sec, minimum=1),
            build_timeout_sec=_int(raw, "build_timeout_sec", defaults.build_timeout_sec, minimum=1),
            cpus=_int(raw, "cpus", defaults.cpus, minimum=1),
            memory_mb=_int(raw, "memory_mb", defaults.memory_mb, minimum=256),
            storage_mb=_int(raw, "storage_mb", defaults.storage_mb, minimum=1),
            max_retries=_int(raw, "max_retries", defaults.max_retries),
            max_context_files=_int(raw, "max_context_files", defaults.max_context_files, minimum=1),
            max_context_chars=_int(raw, "max_context_chars", defaults.max_context_chars, minimum=1000),
            prune_unconverged=_bool(raw, "prune_unconverged", defaults.prune_unconverged),
        )

    def to_protocol_dict(self) -> dict[str, int]:
        return {k: getattr(self, k) for k in ("agent_timeout_sec", "verifier_timeout_sec", "build_timeout_sec",
                                              "cpus", "memory_mb", "storage_mb")}


@dataclass(frozen=True)
class Author:
    name: str
    email: str = ""

    @classmethod
    def from_value(cls, raw: Any) -> "Author":
        if isinstance(raw, str) and raw.strip():
            return cls(name=raw.strip())
        if isinstance(raw, dict) and isinstance(raw.get("name"), str) and raw["name"].strip():
            email = raw.get("email", "")
            return cls(name=raw["name"].strip(), email=str(email).strip() if email else "")
        raise ConfigError("author must be a non-empty string or {name, email}")

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "email": self.email}


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
    author: Author
    seed: int
    protocol_version: str = "1.0"
    difficulty: str = "medium"
    language: str = "en"                 # language of instruction.md (ISO 639-1)
    source: str = ""
    team: str = ""
    limits: Limits = field(default_factory=Limits)
    llm: LLMSettings = field(default_factory=LLMSettings)
    untrusted_dirs: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)

    REQUIRED = ("case_id", "brief", "repository", "limits", "author", "seed")

    @property
    def image_tag(self) -> str:
        slug = re.sub(r"[^a-z0-9._-]+", "-", self.case_id.lower()).strip("-.")
        return f"harness-{slug}:{self.seed}"

    @property
    def language_name(self) -> str:
        return _LANG_NAMES.get(self.language.lower(), self.language)

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

        protocol_version = str(raw.get("protocol_version", "1.0"))
        if not protocol_version.startswith("1."):
            raise ConfigError(f"unsupported protocol_version: {protocol_version}")

        case_id = raw["case_id"]
        if not isinstance(case_id, str) or not _CASE_ID.match(case_id.strip()) or ".." in case_id:
            raise ConfigError("case_id must match [A-Za-z0-9._/-] (e.g. hackathon/settlement-001)")
        brief = _load_brief(raw["brief"], base_dir)
        author = Author.from_value(raw["author"])
        seed = raw["seed"]
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ConfigError("seed must be an integer")
        difficulty = str(raw.get("difficulty", "medium")).strip().lower()
        if difficulty not in ("easy", "medium", "hard"):
            raise ConfigError("difficulty must be easy | medium | hard")
        language = str(raw.get("language", "en")).strip().lower()
        if not re.match(r"^[a-z]{2}(-[a-z]{2})?$", language):
            raise ConfigError("language must be an ISO 639-1 code like 'ru' or 'en'")

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
            author=author,
            seed=seed,
            protocol_version=protocol_version,
            difficulty=difficulty,
            language=language,
            source=str(raw.get("source", "") or ""),
            team=str(raw.get("team", "") or ""),
            limits=Limits.from_dict(raw.get("limits")),
            llm=LLMSettings.from_dict(raw.get("llm"), overrides=llm_overrides),
            untrusted_dirs=untrusted,
            raw=raw,
        )
