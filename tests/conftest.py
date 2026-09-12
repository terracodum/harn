import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DEMO_REPO = ROOT / "examples" / "demo_repo"
DEMO_INPUT = ROOT / "examples" / "demo" / "input.json"
DEMO_MOCK = ROOT / "examples" / "demo" / "mock_responses.json"


@pytest.fixture
def demo_repo_copy(tmp_path: Path) -> Path:
    dest = tmp_path / "repo"
    shutil.copytree(DEMO_REPO, dest, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    return dest


@pytest.fixture
def demo_input(tmp_path: Path, demo_repo_copy: Path) -> Path:
    raw = json.loads(DEMO_INPUT.read_text(encoding="utf-8"))
    raw["repository"] = str(demo_repo_copy)
    raw["output_dir"] = str(tmp_path / "out")
    path = tmp_path / "input.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def mock_responses() -> dict:
    return json.loads(DEMO_MOCK.read_text(encoding="utf-8"))
