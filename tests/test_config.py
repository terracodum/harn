import json

import pytest

from harness.core.config import CaseConfig, ConfigError, LLMSettings


def test_load_demo_input(demo_input, demo_repo_copy):
    cfg = CaseConfig.load(demo_input)
    assert cfg.case_id == "ledger-refund-netting"
    assert cfg.repository == demo_repo_copy.resolve()
    assert cfg.limits.max_retries == 2
    assert cfg.untrusted_dirs == ("tickets",)
    assert cfg.llm.model == "gpt-oss:120b"
    assert cfg.llm.base_url == "http://localhost:11434/v1"


def test_missing_required_key(demo_input):
    raw = json.loads(demo_input.read_text(encoding="utf-8"))
    del raw["author"]
    demo_input.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="author"):
        CaseConfig.load(demo_input)


def test_output_dir_must_be_empty(demo_input, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "junk").write_text("x")
    with pytest.raises(ConfigError, match="not empty"):
        CaseConfig.load(demo_input)


def test_output_dir_inside_repo_rejected(demo_input, demo_repo_copy):
    with pytest.raises(ConfigError, match="inside"):
        CaseConfig.load(demo_input, output_dir=demo_repo_copy / "out")


def test_cli_overrides_win_over_json(demo_input):
    cfg = CaseConfig.load(demo_input, llm_overrides={"model": "qwen3:32b", "base_url": None})
    assert cfg.llm.model == "qwen3:32b"
    assert cfg.llm.base_url == "http://localhost:11434/v1"


def test_env_fallback(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://gigachat-proxy:8000/v1")
    monkeypatch.setenv("LLM_MODEL", "GigaChat-Pro")
    s = LLMSettings.from_dict(None)
    assert s.model == "GigaChat-Pro"
    assert s.base_url == "http://gigachat-proxy:8000/v1"
