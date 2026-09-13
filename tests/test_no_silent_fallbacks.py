"""Contract: an LLM step that fails is reported, never replaced by a heuristic result.

The allow-list below is the complete set of broad `except` blocks in the harness package; add an
entry only together with a justification in the code comment next to it."""
import re
import sys
from pathlib import Path

import pytest

from harness.core.brief import analyze_brief
from harness.core.config import CaseConfig, LLMSettings
from harness.core.errors import PipelineError
from harness.core.llm.base_client import LLMError
from harness.core.llm.mock_client import MockLLMClient
from harness.core.pipeline import Pipeline
from harness.localization.code_retriever import build_index, build_project_tree, localize

ROOT = Path(__file__).resolve().parent.parent / "harness"

ALLOWED_BROAD_EXCEPTS = {
    "core/pipeline.py": 1,                 # turns any crash into result.json (status=failed + error), nothing continues
    "localization/code_retriever.py": 2,   # zvec: release the rocksdb LOCK before re-raising / ignore close() errors
}


def test_no_unlisted_broad_except_blocks():
    found: dict[str, int] = {}
    for path in ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        n = len(re.findall(r"^\s*except(?:\s+(?:Exception|BaseException))?\s*(?:as\s+\w+)?\s*:", text, re.M))
        if n:
            found[path.relative_to(ROOT).as_posix()] = n
    assert found == ALLOWED_BROAD_EXCEPTS, found


def test_every_except_after_an_llm_call_re_raises():
    """An `except` that follows a `complete_json(` call must end in `raise` (a bounded retry is
    fine, a silent continuation is not)."""
    offenders = []
    for path in ROOT.rglob("*.py"):
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if "complete_json(" not in line or "def complete_json" in line:
                continue
            for j in range(i, min(i + 12, len(lines))):
                if re.match(r"\s*except\b", lines[j]):
                    indent = len(lines[j]) - len(lines[j].lstrip())
                    body = []
                    for k in range(j + 1, len(lines)):
                        if lines[k].strip() and (len(lines[k]) - len(lines[k].lstrip())) <= indent:
                            break
                        body.append(lines[k])
                    if not any(re.match(r"\s*raise\b", b) for b in body):
                        offenders.append(f"{path.name}:{j + 1}")
    assert offenders == []


def test_brief_analysis_failure_stops_the_pipeline(demo_input, mock_responses):
    cfg = CaseConfig.load(demo_input)
    llm = MockLLMClient({k: v for k, v in mock_responses.items() if k != "analyze_brief"})
    result = Pipeline(cfg, llm, search_backend="keyword", skip_docker=True).run()
    assert result["status"] == "failed"
    assert result["failed_stage"] == "2a-brief-analysis"
    assert "brief analysis failed" in result["error"] and "mock has no response" in result["error"]
    assert result["cases"] == [] and not (cfg.output_dir / "cases").exists()


def test_synthesis_failure_stops_the_pipeline(demo_input, mock_responses):
    cfg = CaseConfig.load(demo_input)
    llm = MockLLMClient({k: v for k, v in mock_responses.items() if k != "synthesize"})
    result = Pipeline(cfg, llm, search_backend="keyword", skip_docker=True).run()
    assert result["status"] == "failed"
    case = result["cases"][0]
    assert case["status"] == "case_failed" and case["failed_stage"] == "synthesis"
    assert case["error"].startswith("LLM step failed (synthesis)") and "mock has no response" in case["error"]
    assert any("case R1" in lim and "LLM step failed" in lim for lim in result["limitations"])


def test_analyze_brief_never_degrades():
    with pytest.raises(PipelineError):
        analyze_brief(MockLLMClient({}), "some brief that is long enough to be meaningful", [], retries=0)


def test_query_expansion_failure_propagates(demo_repo_copy, tmp_path):
    tree = build_project_tree(demo_repo_copy, untrusted_dirs=["tickets"])
    with pytest.raises(LLMError):
        localize(tree, "refund netting brief", llm=MockLLMClient({}), index_dir=tmp_path / "idx",
                 embedder=None, backend="keyword")


def test_search_backend_is_never_substituted(monkeypatch):
    monkeypatch.setitem(sys.modules, "zvec", None)      # makes `import zvec` fail
    with pytest.raises(RuntimeError, match="zvec package is not importable"):
        build_index([], index_dir=Path("unused"), backend="zvec")
    with pytest.raises(ValueError, match="unknown search backend"):
        build_index([], index_dir=Path("unused"), backend="auto")


def test_openai_client_does_not_switch_mode_on_unrelated_400(monkeypatch):
    openai = pytest.importorskip("openai")
    import importlib
    import importlib.util
    httpx = importlib.import_module("httpx2" if importlib.util.find_spec("httpx2") else "httpx")

    from harness.core.llm.openai_client import OpenAICompatClient

    client = OpenAICompatClient(LLMSettings(model="m", base_url="http://localhost:1/v1"))
    calls: list[str] = []

    def make_error(message: str):
        resp = httpx.Response(400, request=httpx.Request("POST", "http://localhost:1/v1/chat/completions"))
        return openai.BadRequestError(message, response=resp, body=None)

    def fake_call(messages, mode, purpose, schema, max_tokens):
        calls.append(mode)
        raise make_error("This model's maximum context length is 8192 tokens")

    monkeypatch.setattr(client, "_call", fake_call)
    with pytest.raises(LLMError, match="HTTP 400"):
        client.complete_json(purpose="p", system="s", user="u", schema={"required": []})
    assert calls == ["json_schema"]       # no second request under a weaker response_format
    assert client.tracker.calls[-1].ok is False and client.tracker.calls[-1].mode == "json_schema"

    # a 400 that names response_format is a legitimate capability probe: logged and skipped
    calls.clear()

    def fake_call2(messages, mode, purpose, schema, max_tokens):
        calls.append(mode)
        if mode == "json_schema":
            raise make_error("response_format json_schema is not supported")

        class R:
            usage = None
            model = "m"
            choices = [type("C", (), {"finish_reason": "stop", "message": type("M", (), {"content": '{"a": 1}'})()})()]
        return '{"a": 1}', R()

    client2 = OpenAICompatClient(LLMSettings(model="m", base_url="http://localhost:1/v1"))
    monkeypatch.setattr(client2, "_call", fake_call2)
    assert client2.complete_json(purpose="p", system="s", user="u", schema={"required": ["a"]}) == {"a": 1}
    assert calls == ["json_schema", "json_object"]
    assert client2.tracker.calls[-1].mode == "json_object"
