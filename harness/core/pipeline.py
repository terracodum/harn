"""Orchestrator of Stages 1-6."""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Any

from harness.core.artifacts import write_json, write_result, write_task_toml
from harness.core.brief import analyze_brief, reclassify_from_base_run
from harness.core.config import CaseConfig
from harness.core.llm.base_client import BaseLLMClient
from harness.core.snapshot import snapshot_sha256
from harness.core.synthesis import Synthesis, SynthesisEngine, SynthesisError, materialize
from harness.localization.code_retriever import ContextPackage, build_project_tree, localize
from harness.providers.python.detector import PythonStackDetector
from harness.providers.python.env_builder import PythonEnvironmentBuilder
from harness.providers.python.test_runner import PytestJUnitRunner
from harness.validation.convergence import (ConvergenceVerifier, VerificationReport, auto_recategorize,
                                            solution_change_justified)
from harness.validation.docker_runner import DockerRunner

log = logging.getLogger("harness")


class PipelineError(RuntimeError):
    pass


class Pipeline:
    def __init__(self, config: CaseConfig, llm: BaseLLMClient, *, docker: DockerRunner | None = None,
                 embedder: Any | None = None, search_backend: str = "auto", skip_docker: bool = False,
                 isolated_runs: bool = True, keep_image: bool = False) -> None:
        self.config = config
        self.llm = llm
        self.docker = docker or DockerRunner()
        self.embedder = embedder
        self.search_backend = search_backend
        self.skip_docker = skip_docker
        self.isolated_runs = isolated_runs
        self.keep_image = keep_image

        self.output_dir = config.output_dir
        self.task_dir = self.output_dir / "task"
        self.evidence_dir = self.output_dir / "evidence"
        self.work_dir = self.output_dir / ".work"
        self.limitations: list[str] = []
        self.stage_times: dict[str, float] = {}

    # ------------------------------------------------------------------ utils
    def _stage(self, name: str):
        pipeline = self

        class _Timer:
            def __enter__(self):
                log.info("=== %s", name)
                self.t = time.monotonic()
                return self

            def __exit__(self, *exc):
                pipeline.stage_times[name] = round(time.monotonic() - self.t, 3)
                return False
        return _Timer()

    def _archive_attempt(self, n: int) -> None:
        dest = self.evidence_dir / "attempts" / f"{n:02d}"
        dest.mkdir(parents=True, exist_ok=True)
        for item in ("base", "oracle", "isolated", "build.log"):
            src = self.evidence_dir / item
            if src.exists():
                shutil.move(str(src), str(dest / item))
        for item in ("tests", "solution", "instruction.md"):
            src = self.task_dir / item
            if src.exists():
                if src.is_dir():
                    shutil.copytree(src, dest / "task" / item)
                else:
                    (dest / "task").mkdir(exist_ok=True)
                    shutil.copy2(src, dest / "task" / item)

    # -------------------------------------------------------------------- run
    def run(self) -> dict[str, Any]:
        cfg = self.config
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.task_dir.mkdir(exist_ok=True)
        self.evidence_dir.mkdir(exist_ok=True)
        status, error, attempts = "failed", None, 0
        manifest: dict[str, list[str]] = {"fail_to_pass": [], "pass_to_pass": [], "anti_cheat": []}
        report: VerificationReport | None = None
        snapshot_before = ""
        engine: SynthesisEngine | None = None
        try:
            with self._stage("1-discovery"):
                snapshot_before = snapshot_sha256(cfg.repository)
                tree = build_project_tree(cfg.repository, untrusted_dirs=cfg.untrusted_dirs)
                profile = PythonStackDetector().detect(cfg.repository, tree)
                write_json(self.evidence_dir / "profile.json", {
                    "snapshot_sha256": snapshot_before, "files": len(tree.files),
                    "untrusted_files": [f.rel_path for f in tree.files if not f.trusted][:200],
                    "profile": profile.to_dict()})
                log.info("stack: %s", profile.to_dict())

            with self._stage("2a-brief-analysis"):
                spec = analyze_brief(self.llm, cfg.brief, [f.rel_path for f in tree.trusted_files])
                write_json(self.evidence_dir / "brief_spec.json", spec.to_dict())
                log.info("requirements: %s", [f"{r.id}:{r.title[:60]}" for r in spec.requirements])
                for a in spec.assumptions:
                    self.limitations.append(f"assumption: {a}")
                for q in spec.ambiguities:
                    self.limitations.append(f"ambiguity: {q}")

            with self._stage("2b-localization"):
                ctx: ContextPackage = localize(
                    tree, cfg.brief, spec=spec, index_dir=self.work_dir / "index", embedder=self.embedder,
                    backend=self.search_backend, max_files=cfg.limits.max_context_files,
                    max_chars=cfg.limits.max_context_chars)
                write_json(self.evidence_dir / "localization.json", ctx.summary())
                log.info("context files: %s", [f.path for f in ctx.files])

            engine = SynthesisEngine(self.llm, brief=cfg.brief, profile=profile, spec=spec,
                                     instruction_language=cfg.language_name, difficulty=cfg.difficulty)
            env_builder = PythonEnvironmentBuilder()
            with self._stage("3-synthesis"):
                syn: Synthesis = engine.synthesize(ctx, max_chars=cfg.limits.max_context_chars)
                materialize(self.task_dir, syn)
                manifest = syn.manifest
                write_json(self.evidence_dir / "synthesis.json", {"root_cause": syn.root_cause, "notes": syn.notes,
                                                                   "manifest": syn.manifest, "coverage": syn.coverage,
                                                                   "extra_pip_packages": syn.extra_pip_packages})

            with self._stage("4-environment"):
                env_builder.build(task_dir=self.task_dir, repo=cfg.repository, tree=tree, profile=profile,
                                  extra_packages=syn.extra_pip_packages)

            if self.skip_docker:
                status = "unverified"
                self.limitations.append("verification skipped (--skip-docker): Base/Oracle runs not executed")
            else:
                ok, info = self.docker.available()
                if not ok:
                    raise PipelineError(f"docker is not available: {info}")
                image_tag = cfg.image_tag
                verifier = ConvergenceVerifier(self.docker, PytestJUnitRunner(), task_dir=self.task_dir,
                                               evidence_dir=self.evidence_dir, limits=cfg.limits,
                                               image_tag=image_tag, isolated_runs=self.isolated_runs)
                need_build = True
                packages = list(syn.extra_pip_packages)
                with self._stage("5-convergence"):
                    heal_rejection: str | None = None   # validator verdict on the last healed bundle
                    for attempt in range(cfg.limits.max_retries + 1):
                        attempts = attempt + 1
                        if heal_rejection is None:
                            log.info("verification attempt %d/%d", attempts, cfg.limits.max_retries + 1)
                            report = verifier.verify(manifest, need_build=need_build)
                            if not report.build_ok:
                                # apt/pip mirrors flake: one immediate rebuild before judging the bundle
                                log.warning("docker build failed, retrying the build once")
                                shutil.copy2(self.evidence_dir / "build.log", self.evidence_dir / "build.first-try.log")
                                report = verifier.verify(manifest, need_build=True)
                            write_json(self.evidence_dir / "verification.json", report.to_dict())
                            if report.ok:
                                break
                            log.warning("verification failed: %s", " | ".join(p[:200] for p in report.problems))
                            if not report.build_ok and not packages:
                                # nothing in the bundle influences the image -> healing cannot help
                                raise PipelineError("environment image failed to build (see evidence/build.log)")
                            self._archive_attempt(attempts)
                            moved = auto_recategorize(report, syn.manifest)
                            if moved:
                                log.info("auto-recategorized %d leaky fail_to_pass test(s) to pass_to_pass: %s", len(moved), moved)
                                reclassify_from_base_run(spec, syn.coverage, report.base.table)
                                write_json(self.evidence_dir / "brief_spec.json", spec.to_dict())
                                materialize(self.task_dir, syn)
                                manifest = syn.manifest
                                write_json(self.evidence_dir / "synthesis.json", {"root_cause": syn.root_cause, "notes": syn.notes,
                                                                                   "manifest": syn.manifest, "coverage": syn.coverage,
                                                                                   "extra_pip_packages": syn.extra_pip_packages,
                                                                                   "auto_recategorized": moved})
                                need_build = False
                                continue    # re-verify without spending an LLM round
                        else:
                            # the previous healed bundle never reached the sandbox: re-verifying the old one is
                            # pointless, ask the model again with the validator's verdict attached
                            log.info("healing attempt %d/%d (bundle rejected by validator, no sandbox run)",
                                     attempts, cfg.limits.max_retries + 1)
                        if attempt == cfg.limits.max_retries:
                            break
                        problems = list(report.problems)
                        if heal_rejection:
                            problems.append(f"your previous corrected bundle was rejected before execution: {heal_rejection}")
                        if report.base is not None:
                            notes = reclassify_from_base_run(spec, syn.coverage, report.base.table)
                            if notes:
                                log.info("reclassified requirements: %s", notes)
                                write_json(self.evidence_dir / "brief_spec.json", spec.to_dict())
                                problems = notes + problems
                        try:
                            healed = engine.heal(ctx, syn, problems, report.logs, max_chars=cfg.limits.max_context_chars)
                        except SynthesisError as exc:
                            log.warning("healing produced an invalid bundle: %s", exc)
                            heal_rejection = str(exc)[:2000]
                            continue
                        if healed.solve_sh.strip() != syn.solve_sh.strip() and not solution_change_justified(report):
                            # guard against "fixing" a wrong test by bending the reference solution
                            heal_rejection = ("solve.sh was modified although every failure was on the BASE run "
                                              "(pass_to_pass / anti_cheat failing on the original code, or fail_to_pass "
                                              "passing on it). Those are TEST defects: keep solve.sh exactly as it was "
                                              "and fix the tests / categorisation instead.")
                            log.warning("healed bundle rejected: unjustified solve.sh change")
                            continue
                        syn = healed
                        heal_rejection = None
                        materialize(self.task_dir, syn)
                        manifest = syn.manifest
                        write_json(self.evidence_dir / "synthesis.json", {"root_cause": syn.root_cause, "notes": syn.notes,
                                                                           "manifest": syn.manifest, "coverage": syn.coverage,
                                                                           "extra_pip_packages": syn.extra_pip_packages,
                                                                           "healed_attempt": attempts})
                        need_build = (not report.build_ok) or syn.extra_pip_packages != packages
                        packages = list(syn.extra_pip_packages)
                        if need_build:
                            env_builder.build(task_dir=self.task_dir, repo=cfg.repository, tree=tree, profile=profile,
                                              extra_packages=packages)
                    if report and report.ok:
                        status = "ready"
                    else:
                        error = "convergence failed: " + " | ".join(p[:300] for p in (report.problems if report else ["no report"])[:10])
                    if not self.keep_image:
                        self.docker.remove_image(image_tag)

            with self._stage("6-packaging"):
                snapshot_after = snapshot_sha256(cfg.repository)
                if snapshot_after != snapshot_before:
                    status, error = "failed", "source repository changed during generation"
                write_task_toml(self.task_dir, cfg, profile, manifest, snapshot_before)
                for note in profile.notes:
                    self.limitations.append(f"stack: {note}")
                if profile.uses_postgres:
                    self.limitations.append("PostgreSQL provisioned with a default cluster; only `alembic upgrade head` is applied")
        except (PipelineError, SynthesisError) as exc:
            error = str(exc)
            log.error("pipeline failed: %s", exc)
        except Exception as exc:  # noqa: BLE001 - result.json must always be written
            error = f"{type(exc).__name__}: {exc}"
            log.exception("pipeline crashed")
        finally:
            self.llm.tracker.write(self.evidence_dir / "llm_usage.json")
            if engine is not None:
                for i, item in enumerate(engine.raw_history, 1):
                    write_json(self.evidence_dir / "llm_responses" / f"{i:02d}_{item['purpose']}.json", item["data"])
            write_json(self.evidence_dir / "summary.json", {
                "status": status, "attempts": attempts, "stage_times_sec": self.stage_times,
                "runs": report.runs if report else [],
                "isolated": report.isolated if report else {},
                "problems": report.problems if report else ([error] if error else []),
            })
            shutil.rmtree(self.work_dir, ignore_errors=True)
            result = write_result(self.output_dir, status=status, error=error, limitations=self.limitations,
                                  attempts=attempts, extra={"case_id": cfg.case_id,
                                                            "input_snapshot_sha256": snapshot_before,
                                                            "llm": self.llm.tracker.summary() | {"calls": None}})
        return result
