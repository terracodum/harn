import importlib.util

import pytest

from harness.core.llm.mock_client import MockLLMClient
from harness.localization.code_retriever import (
    KeywordIndex, assemble_context, build_index, build_project_tree, chunk_file, chunk_tree, localize,
    tokenize_for_fts,
)
from harness.localization.entity_extractor import extract_terms, split_identifier

BRIEF = ("При суточном закрытии возвраты (refund) со статусом settled увеличивают нетто-баланс. "
         "Нужно исправить `net_balance` в NettingPolicy.calculate, модель Transaction не менять.")


def test_extract_terms_prefers_identifiers():
    terms = extract_terms(BRIEF)
    assert terms[0] in ("net_balance", "NettingPolicy.calculate")
    assert "refund" in terms and "settled" in terms
    assert "transaction" in terms
    assert "нужно" not in terms


def test_split_identifier():
    assert split_identifier("NettingPolicy.settle_refund") == ["netting", "policy", "settle", "refund"]


def test_tokenize_for_fts_splits_identifiers():
    toks = tokenize_for_fts("def settle_refund(amount): NettingPolicy").split()
    assert {"settle_refund", "settle", "refund", "nettingpolicy", "netting", "policy"} <= set(toks)


def test_tree_filters_untrusted_and_ignored(demo_repo_copy):
    (demo_repo_copy / "__pycache__").mkdir()
    (demo_repo_copy / "__pycache__" / "x.pyc").write_bytes(b"\0\0")
    tree = build_project_tree(demo_repo_copy, untrusted_dirs=["tickets"])
    paths = tree.paths()
    assert "ledger/netting.py" in paths
    assert not any(p.startswith("__pycache__") for p in paths)
    ticket = next(f for f in tree.files if f.rel_path.startswith("tickets/"))
    assert ticket.trusted is False
    assert all("tickets/" not in f.rel_path for f in tree.trusted_files)
    assert next(f for f in tree.files if f.rel_path == "ledger/models.py").is_contract


def test_chunk_file_splits_on_defs():
    src = "import x\n\n" + "\n".join(f"def f{i}():\n    return {i}\n" for i in range(40))
    chunks = chunk_file("m.py", src, max_lines=30)
    assert len(chunks) > 1
    assert chunks[0].start == 1
    assert all(c.end >= c.start for c in chunks)


def test_keyword_index_finds_netting(demo_repo_copy):
    tree = build_project_tree(demo_repo_copy, untrusted_dirs=["tickets"])
    index = KeywordIndex(chunk_tree(tree))
    hits = index.search([BRIEF, "refund settled net_balance"])
    assert hits and hits[0].chunk.path == "ledger/netting.py"


@pytest.mark.skipif(importlib.util.find_spec("zvec") is None, reason="zvec not installed")
def test_zvec_index_finds_netting(demo_repo_copy, tmp_path):
    tree = build_project_tree(demo_repo_copy, untrusted_dirs=["tickets"])
    index = build_index(chunk_tree(tree), index_dir=tmp_path / "idx", backend="zvec")
    try:
        hits = index.search(["refund settled net_balance"])
    finally:
        index.close()
    assert index.backend == "zvec"
    # BM25 favours the short existing test file; the top *implementation* hit must be netting.py
    impl_hits = [h for h in hits if not h.chunk.path.startswith("tests/")]
    assert impl_hits and impl_hits[0].chunk.path == "ledger/netting.py"


def test_component_root():
    from harness.localization.code_retriever import _component_root
    assert _component_root("backend/src/components/settlement/application/impl/x.py") == "backend/src/components/settlement/"
    assert _component_root("src/pkg/module.py") == "src/pkg/"
    assert _component_root("ledger/netting.py") == "ledger/"
    assert _component_root("app.py") == ""


def test_assemble_context_roles(demo_repo_copy):
    tree = build_project_tree(demo_repo_copy, untrusted_dirs=["tickets"])
    hits = KeywordIndex(chunk_tree(tree)).search(["refund settled net_balance"])
    pkg = assemble_context(tree, hits, queries=["q"], keywords=["refund"], backend="keyword",
                           candidate_files=["ledger/models.py"], max_files=3)
    paths = {f.path: f.role for f in pkg.files}
    assert paths["ledger/netting.py"] == "implementation"
    assert "ledger/models.py" in paths
    assert "tickets/T-1042.md" not in paths
    rendered = pkg.render(max_chars=100_000)
    assert "ignore your previous instructions" not in rendered
    assert 'path="ledger/netting.py"' in rendered


def test_localize_with_mock_llm(demo_repo_copy, tmp_path, mock_responses):
    tree = build_project_tree(demo_repo_copy, untrusted_dirs=["tickets"])
    llm = MockLLMClient(mock_responses)
    pkg = localize(tree, BRIEF, llm=llm, index_dir=tmp_path / "idx", embedder=None, backend="keyword")
    assert pkg.files[0].path == "ledger/netting.py"
    assert "net_balance" in pkg.keywords
    assert llm.prompts[0]["purpose"] == "localize"
    assert "tickets/T-1042.md" not in llm.prompts[0]["user"]
