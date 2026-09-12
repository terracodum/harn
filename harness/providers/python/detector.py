"""Python stack detection (Stage 1.2 / 1.3)."""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

from harness.localization.code_retriever import ProjectTree, read_text
from harness.providers.base import IStackDetector, StackProfile

_PG_MARKERS = re.compile(r"psycopg|asyncpg|postgres|pg8000", re.IGNORECASE)
_SQLITE_MARKERS = re.compile(r"\bsqlite3?\b|aiosqlite", re.IGNORECASE)
_PY_REQ = re.compile(r"(\d+)\.(\d+)")
_ENV_READ = re.compile(r"(?:os\.environ(?:\.get)?\s*[\[(]\s*|os\.getenv\s*\(\s*|\$\{?)[\"']?([A-Z][A-Z0-9_]{2,})")


class PythonStackDetector(IStackDetector):
    def detect(self, repo: Path, tree: ProjectTree) -> StackProfile:
        profile = StackProfile(language="python")
        paths = set(tree.paths())

        for name in ("pyproject.toml", "setup.py", "setup.cfg", "Pipfile", "requirements.txt",
                     "requirements.lock", "requirements-dev.txt", "requirements_dev.txt", "dev-requirements.txt"):
            if name in paths:
                profile.manifests.append(name)
        for p in sorted(paths):
            if re.match(r"^requirements[^/]*\.(txt|lock)$", p):
                profile.requirements_files.append(p)

        dep_text = ""
        for p in profile.requirements_files + [m for m in profile.manifests if m in ("pyproject.toml", "setup.py", "setup.cfg", "Pipfile")]:
            dep_text += (read_text(repo / p) or "") + "\n"

        if "pyproject.toml" in paths:
            try:
                data = tomllib.loads(read_text(repo / "pyproject.toml") or "")
            except tomllib.TOMLDecodeError:
                data = {}
                profile.notes.append("pyproject.toml is not valid TOML")
            project = data.get("project") or {}
            tool = data.get("tool") or {}
            # `pip install -e .` only makes sense with a declared build backend (or poetry)
            if "build-system" in data or tool.get("poetry"):
                profile.installable = True
            pytest_cfg = tool.get("pytest", {}).get("ini_options", {}) if isinstance(tool.get("pytest"), dict) else {}
            for extra in pytest_cfg.get("pythonpath", []) or []:
                extra = str(extra).strip("./")
                if extra and extra != "." and extra not in profile.package_dirs:
                    profile.package_dirs.append(extra)
            req = str(project.get("requires-python", "")) or str(((data.get("tool") or {}).get("poetry") or {}).get("dependencies", {}).get("python", ""))
            m = _PY_REQ.search(req)
            if m and int(m.group(1)) == 3 and int(m.group(2)) >= 11:
                profile.python_version = f"3.{m.group(2)}"
            deps = project.get("dependencies") or []
            optional = project.get("optional-dependencies") or {}
            dep_text += "\n".join(map(str, deps)) + "\n" + "\n".join(str(x) for v in optional.values() for x in v)
            if (data.get("tool") or {}).get("pytest"):
                profile.test_framework = "pytest"
        elif "setup.py" in paths:
            profile.installable = True

        if "src" in {p.split("/")[0] for p in paths} and any(p.startswith("src/") and p.endswith(".py") for p in paths):
            if "src" not in profile.package_dirs:
                profile.package_dirs.append("src")
        if "alembic.ini" in paths:
            m = re.search(r"^prepend_sys_path\s*=\s*(.+)$", read_text(repo / "alembic.ini") or "", re.MULTILINE)
            if m:
                extra = m.group(1).strip().replace("%(here)s", "").strip("/ .")
                if extra and extra not in profile.package_dirs:
                    profile.package_dirs.append(extra)

        # tests
        test_dirs = sorted({p.split("/")[0] for p in paths if p.split("/")[0] in ("tests", "test")})
        profile.test_dirs = test_dirs
        test_files = [p for p in paths if re.search(r"(^|/)test_[^/]*\.py$|_test\.py$", p)]
        if profile.test_framework is None and test_files:
            sample = "\n".join((read_text(repo / p) or "")[:4000] for p in test_files[:10])
            if "import pytest" in sample or "pytest" in dep_text or "conftest.py" in " ".join(paths):
                profile.test_framework = "pytest"
            elif "unittest" in sample:
                profile.test_framework = "unittest"
            else:
                profile.test_framework = "pytest"

        # infrastructure
        profile.alembic = "alembic.ini" in paths or any(p.startswith("alembic/") or p.startswith("migrations/") for p in paths)
        profile.sql_files = sorted(p for p in paths if p.endswith(".sql"))[:50]
        profile.seed_files = sorted(p for p in paths if p.endswith(".sql") and "seed" in p.lower()
                                    and not p.startswith(("tests/", "test/")))
        env_text = "\n".join(read_text(repo / p) or "" for p in paths if p.endswith((".env", ".env.example", "alembic.ini")))
        profile.uses_postgres = bool(_PG_MARKERS.search(dep_text + env_text))
        if profile.uses_postgres and re.search(r"sqlalchemy|alembic", dep_text, re.IGNORECASE):
            if re.search(r"psycopg2", dep_text, re.IGNORECASE):
                profile.db_url_scheme = "postgresql+psycopg2"
            elif re.search(r"psycopg", dep_text, re.IGNORECASE):
                profile.db_url_scheme = "postgresql+psycopg"
            elif re.search(r"asyncpg", dep_text, re.IGNORECASE):
                profile.db_url_scheme = "postgresql+asyncpg"
        code_sample = ""
        for p in [p for p in paths if p.endswith(".py")][:200]:
            code_sample += (read_text(repo / p) or "")[:2000]
        profile.uses_sqlite = bool(_SQLITE_MARKERS.search(dep_text + code_sample)) and not profile.uses_postgres
        # environment variables the project reads (so the sandbox can provide DB ones)
        names: set[str] = set()
        for p in [p for p in paths if p.endswith((".py", ".ini", ".cfg", ".toml", ".md", ".env.example"))][:400]:
            for m in _ENV_READ.finditer(read_text(repo / p) or ""):
                names.add(m.group(1))
        profile.env_vars = sorted(names)
        if profile.uses_postgres:
            for name in profile.env_vars:
                upper = name.upper()
                if not re.search(r"DSN|DATABASE|POSTGRES|PG_|DB_URL|DB_URI", upper):
                    continue
                if "DSN" in upper and "URL" not in upper:
                    profile.db_env[name] = "host=localhost port=5432 dbname=harness user=harness password=harness"
                else:
                    profile.db_env[name] = f"{profile.db_url_scheme}://harness:harness@localhost:5432/harness"
        if not profile.manifests:
            profile.notes.append("no dependency manifest found; only pytest will be installed")
        return profile
