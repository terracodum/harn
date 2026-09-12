"""CLI: python -m harness run input.json --output-dir out/"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from harness.core.config import CaseConfig, ConfigError
from harness.core.llm.token_tracker import TokenTracker


def _build_llm(args: argparse.Namespace, config: CaseConfig, tracker: TokenTracker):
    if args.mock_llm:
        from harness.core.llm.mock_client import MockLLMClient
        return MockLLMClient.from_file(args.mock_llm, tracker)
    from harness.core.llm.openai_client import OpenAICompatClient
    return OpenAICompatClient(config.llm, tracker)


def cmd_run(args: argparse.Namespace) -> int:
    overrides = {"model": args.model, "base_url": args.base_url, "embedding_model": args.embedding_model}
    try:
        config = CaseConfig.load(args.input, output_dir=args.output_dir, llm_overrides=overrides)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    tracker = TokenTracker()
    llm = _build_llm(args, config, tracker)
    embedder = None
    if config.llm.embedding_model and not args.mock_llm:
        from harness.localization.embeddings import make_embedder
        embedder = make_embedder(config.llm)

    from harness.core.pipeline import Pipeline
    pipeline = Pipeline(config, llm, embedder=embedder, search_backend=args.search_backend,
                        skip_docker=args.skip_docker, isolated_runs=not args.no_isolated, keep_image=args.keep_image)
    result = pipeline.run()
    print(json.dumps({k: result[k] for k in ("status", "attempts", "error", "limitations")}, indent=2, ensure_ascii=False))
    return 0 if result["status"] in ("ready", "unverified") else 1


def cmd_snapshot(args: argparse.Namespace) -> int:
    from harness.core.snapshot import snapshot_sha256
    print(snapshot_sha256(Path(args.repository)))
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Stages 1-2 only (no synthesis, no docker); prints the profile and the context package."""
    from harness.localization.code_retriever import build_project_tree, localize
    from harness.providers.python.detector import PythonStackDetector

    overrides = {"model": args.model, "base_url": args.base_url, "embedding_model": args.embedding_model}
    try:
        config = CaseConfig.load(args.input, output_dir=args.output_dir or "__inspect__", llm_overrides=overrides)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    tree = build_project_tree(config.repository, untrusted_dirs=config.untrusted_dirs)
    profile = PythonStackDetector().detect(config.repository, tree)
    llm = None if args.no_llm else _build_llm(args, config, TokenTracker())
    embedder = None
    if config.llm.embedding_model and not args.mock_llm:
        from harness.localization.embeddings import make_embedder
        embedder = make_embedder(config.llm)
    from harness.core.brief import analyze_brief
    spec = analyze_brief(llm, config.brief, [f.rel_path for f in tree.trusted_files])
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        ctx = localize(tree, config.brief, spec=spec, index_dir=Path(tmp) / "index", embedder=embedder,
                       backend=args.search_backend, max_files=config.limits.max_context_files,
                       max_chars=config.limits.max_context_chars)
    print(json.dumps({"profile": profile.to_dict(), "brief_spec": spec.to_dict(), "context": ctx.summary()},
                     indent=2, ensure_ascii=False))
    if args.show_context:
        print(ctx.render(max_chars=config.limits.max_context_chars))
    return 0


def _add_llm_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", help="chat model id (e.g. gpt-oss:120b, qwen3:32b, GigaChat-Pro)")
    p.add_argument("--base-url", help="OpenAI-compatible base URL (e.g. http://localhost:11434/v1)")
    p.add_argument("--embedding-model", help="embedding model for zvec dense search (optional)")
    p.add_argument("--mock-llm", help="JSON file with canned responses (offline mode)")
    p.add_argument("--search-backend", choices=["auto", "zvec", "keyword"], default="auto")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="harness", description="Benchmark case generator")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="generate and verify a benchmark case")
    p_run.add_argument("input", help="input JSON")
    p_run.add_argument("--output-dir")
    p_run.add_argument("--skip-docker", action="store_true", help="stop after packaging (no verification)")
    p_run.add_argument("--no-isolated", action="store_true", help="skip per-category isolated runs")
    p_run.add_argument("--keep-image", action="store_true")
    _add_llm_args(p_run)
    p_run.set_defaults(func=cmd_run)

    p_snap = sub.add_parser("snapshot", help="print input_snapshot_sha256 of a directory")
    p_snap.add_argument("repository")
    p_snap.set_defaults(func=cmd_snapshot)

    p_ins = sub.add_parser("inspect", help="run discovery + localization only")
    p_ins.add_argument("input")
    p_ins.add_argument("--output-dir")
    p_ins.add_argument("--no-llm", action="store_true", help="heuristic queries only")
    p_ins.add_argument("--show-context", action="store_true")
    _add_llm_args(p_ins)
    p_ins.set_defaults(func=cmd_inspect)

    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # Cyrillic briefs on Windows consoles
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
