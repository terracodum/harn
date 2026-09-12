import json

import pytest

from harness.core.config import CaseConfig, ConfigError, LLMSettings
from tests.conftest import ROOT


def test_load_demo_input(demo_input, demo_repo_copy):
    cfg = CaseConfig.load(demo_input)
    assert cfg.case_id == "demo/ledger-refund-netting"
    assert cfg.repository == demo_repo_copy.resolve()
    assert cfg.limits.max_retries == 2
    assert cfg.limits.verifier_timeout_sec == 600 and cfg.limits.run_timeout_sec == 600
    assert cfg.untrusted_dirs == ("tickets",)
    assert cfg.author.name == "harness-demo" and cfg.author.email == "demo@example.org"
    assert cfg.language == "ru" and cfg.language_name == "Russian"
    assert cfg.image_tag == "harness-demo-ledger-refund-netting:1"
    assert cfg.llm.model == "gpt-oss:120b"
    assert cfg.llm.base_url == "http://localhost:11434/v1"


def test_protocol_example_format(tmp_path, demo_repo_copy):
    raw = json.loads((ROOT / "examples" / "protocol" / "input.example.json").read_text(encoding="utf-8"))
    raw["repository"] = str(demo_repo_copy)
    raw["output_dir"] = str(tmp_path / "runs" / "example")
    raw["brief"] = "Refunds are added instead of subtracted in the daily close."
    cfg = CaseConfig.from_dict(raw, base_dir=tmp_path)
    assert cfg.protocol_version == "1.0"
    assert cfg.case_id == "hackathon/settlement-001"
    assert cfg.difficulty == "medium" and cfg.language == "ru"
    assert cfg.limits.to_protocol_dict() == {
        "agent_timeout_sec": 1800, "verifier_timeout_sec": 300, "build_timeout_sec": 900,
        "cpus": 2, "memory_mb": 4096, "storage_mb": 10240,
    }
    assert cfg.author.to_dict() == {"name": "Участник Примеров", "email": "participant@example.org"}
    assert cfg.source == "hackathon/settlement-001" and cfg.team == "team-example"
    assert cfg.image_tag == "harness-hackathon-settlement-001:4107"


def test_empty_brief_rejected(tmp_path, demo_repo_copy):
    raw = json.loads((ROOT / "examples" / "protocol" / "input.example.json").read_text(encoding="utf-8"))
    raw["repository"] = str(demo_repo_copy)
    with pytest.raises(ConfigError, match="brief"):
        CaseConfig.from_dict(raw, base_dir=tmp_path)


def test_legacy_string_author_and_run_timeout(demo_input):
    raw = json.loads(demo_input.read_text(encoding="utf-8"))
    raw["author"] = "someone"
    raw["limits"] = {"run_timeout_sec": 42}
    demo_input.write_text(json.dumps(raw), encoding="utf-8")
    cfg = CaseConfig.load(demo_input)
    assert cfg.author.name == "someone" and cfg.author.email == ""
    assert cfg.limits.verifier_timeout_sec == 42


def test_missing_required_key(demo_input):
    raw = json.loads(demo_input.read_text(encoding="utf-8"))
    del raw["author"]
    demo_input.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="author"):
        CaseConfig.load(demo_input)


def test_bad_case_id_rejected(demo_input):
    raw = json.loads(demo_input.read_text(encoding="utf-8"))
    raw["case_id"] = "../evil id"
    demo_input.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="case_id"):
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
