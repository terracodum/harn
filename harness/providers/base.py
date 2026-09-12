"""Stack-dependent provider interfaces (Section 4 of HARNESS_ARCHITECTURE.md)."""
from __future__ import annotations

import abc
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from harness.localization.code_retriever import ProjectTree


@dataclass
class StackProfile:
    language: str
    manifests: list[str] = field(default_factory=list)       # pyproject.toml, requirements*.txt, ...
    requirements_files: list[str] = field(default_factory=list)
    installable: bool = False                                # `pip install -e .` makes sense
    package_dirs: list[str] = field(default_factory=list)    # e.g. ["src"] -> PYTHONPATH extra
    python_version: str = "3.11"
    test_framework: str | None = None                        # pytest | unittest | None
    test_dirs: list[str] = field(default_factory=list)
    uses_postgres: bool = False
    uses_sqlite: bool = False
    db_url_scheme: str = "postgresql"                        # SQLAlchemy-style when the project uses it
    alembic: bool = False
    sql_files: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TestStatus:
    node_id: str
    outcome: str            # passed | failed | error | skipped | missing
    message: str = ""


@dataclass
class RunResult:
    reward: int | None      # None when reward.txt is missing
    tests: dict[str, TestStatus]
    exit_code: int | None
    duration_sec: float
    timed_out: bool = False
    notes: list[str] = field(default_factory=list)


class IStackDetector(abc.ABC):
    @abc.abstractmethod
    def detect(self, repo: Path, tree: ProjectTree) -> StackProfile: ...


class IEnvironmentBuilder(abc.ABC):
    @abc.abstractmethod
    def build(self, *, task_dir: Path, repo: Path, tree: ProjectTree, profile: StackProfile,
              extra_packages: list[str]) -> None:
        """Write task/environment/{Dockerfile,repo/}, task/tests/test.sh and task/tests/verify.py."""


class ITestRunner(abc.ABC):
    @abc.abstractmethod
    def parse_results(self, logs_dir: Path, manifest: dict[str, list[str]], category: str) -> RunResult: ...
