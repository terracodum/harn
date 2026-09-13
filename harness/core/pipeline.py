"""Orchestrator of Stages 1-6.

Failure policy: every LLM step either succeeds or raises (`LLMError`, `SynthesisError`,
`PipelineError`). Nothing here catches a model failure to continue with a heuristic result; the
`except` blocks at the bottom only turn the exception into result.json (status=failed,
failed_stage, error) so that a run always leaves a readable trace.
"""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Any

from harness.core.artifacts import write_json, write_result, write_task_toml
from harness.core.brief import NEEDS_FAIL_TO_PASS, BriefSpec, analyze_brief
from harness.core.config import CaseConfig
from harness.core.errors import PipelineError
from harness.core.llm.base_client import BaseLLMClient, LLMError
from harness.core.snapshot import snapshot_sha256
from harness.core.synthesis import Synthesis, SynthesisEngine, SynthesisError, materialize
from harness.localization.code_retriever import ContextPackage, build_project_tree, localize
from harness.providers.python.detector import PythonStackDetector
from harness.providers.python.env_builder import PythonEnvironmentBuilder
from harness.providers.python.test_runner import PytestJUnitRunner
from harness.validation.convergence import (Attribution, ConvergenceVerifier, VerificationReport,
                                            attribute_problems, solution_change_justified)
from harness.validation.docker_runner import DockerRunner

log = logging.getLogger("harness")

__all__ = ["Pipeline", "PipelineError", "prune_candidates"]


def prune_candidates(attribution: Attribution, spec: BriefSpec, syn: Synthesis) -> tuple[list[str], str | None]:
    """Which requirements may be dropped so that the rest of the case survives.

    Returns (ids, reason_if_impossible). Pruning is refused when a problem cannot be pinned to a
    requirement (it could concern any of them) or when dropping the failing requirements would
    leave no fail_to_pass test / no bug-like requirement - such a case has no value."""
    if attribution.unattributed:
        return [], "problems not attributable to a requirement: " + " | ".join(p[:200] for p in attribution.unattributed[:5])
    failing = [r.id for r in spec.requirements if attribution.by_requirement.get(r.id)]
    if not failing:
        return [], "no requirement-bound problems"
    remaining = [r for r in spec.requirements if r.id not in failing]
    if not any(r.kind in NEEDS_FAIL_TO_PASS and r.testable for r in remaining):
        return [], "pruning would leave no bug/feature/change requirement"
    keep_tests = {t for r in remaining for t in syn.tests_of(r.id)}
    if not keep_tests & set(syn.manifest["fail_to_pass"]):
        return [], "pruning would leave no fail_to_pass test"
    return failing, None


class Pipeline:
    def __init__(self, config: CaseConfig, llm: BaseLLMClient, *, docker: DockerRunner | None = None,
                 embedder: Any | None = None, search_backend: str = "zvec", skip_docker: bool = False,
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
        self.current_stage = "init"

    # ------------------------------------------------------------------ utils
    def _stage(self, name: str):
        pipeline = self

        class _Timer:
            def __enter__(self):
                log.info("=== %s", name)
                pipeline.current_stage = name
                self.t = time.monotonic()
                return self

            def __exit__(self, *exc):
                pipeline.stage_times[name] = round(time.monotonic() - self.t, 3)
                return False
        return _Timer()

    def _archive_attempt(self, n: int) -> None:
        dest = self.evidence_dir / "attempts" / f"{n:02d}"
        dest.mkdir(parents=True, exist_ok=True)
        for item in ("base", "oracle", "isolated", "staged", "build.log"):
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

    def _write_synthesis_evidence(self, syn: Synthesis, **extra: Any) -> None:
        write_json(self.evidence_dir / "synthesis.json", {
            "root_cause": syn.root_cause, "notes": syn.notes, "manifest": syn.manifest, "coverage": syn.coverage,
            "solve_steps": syn.solution.order, "pruned": syn.solution.pruned,
            "extra_pip_packages": syn.extra_pip_packages, **extra})

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
                self._write_synthesis_evidence(syn)
                log.info("solve steps: %s", syn.solution.order)

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
                with self._stage("5-convergence"):
                    report, attempts, syn = self._converge(verifier, env_builder, engine, ctx, spec, syn,
                                                           tree=tree, profile=profile)
                    manifest = syn.manifest
                    if report.ok:
                        status = "ready"
                    else:
                        error = "convergence failed: " + " | ".join(p[:300] for p in report.problems[:10])
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
        except SynthesisError as exc:
            error = f"bundle rejected by the validator before any sandbox run (stage {self.current_stage}): {exc}"
            log.error("pipeline failed: %s", error)
        except LLMError as exc:
            error = f"LLM step failed (stage {self.current_stage}): {exc}"
            log.error("pipeline failed: %s", error)
        except PipelineError as exc:
            error = f"{exc} (stage {self.current_stage})"
            log.error("pipeline failed: %s", error)
        except Exception as exc:  # noqa: BLE001 - result.json must always be written; the error is not swallowed
            error = f"{type(exc).__name__}: {exc} (stage {self.current_stage})"
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
                                                            "failed_stage": None if status != "failed" else self.current_stage,
                                                            "input_snapshot_sha256": snapshot_before,
                                                            "llm": {k: v for k, v in self.llm.tracker.summary().items()
                                                                    if k != "calls"} | {"details": "evidence/llm_usage.json"}})
        return result

    # ------------------------------------------------------------ convergence
    def _verify_once(self, verifier: ConvergenceVerifier, manifest: dict[str, list[str]], *, need_build: bool,
                     packages: list[str]) -> VerificationReport:
        report = verifier.verify(manifest, need_build=need_build)
        if need_build and not report.build_ok:
            # apt/pip mirrors flake: one immediate rebuild before judging the bundle
            log.warning("docker build failed, retrying the build once")
            shutil.copy2(self.evidence_dir / "build.log", self.evidence_dir / "build.first-try.log")
            report = verifier.verify(manifest, need_build=True)
        write_json(self.evidence_dir / "verification.json", report.to_dict())
        if not report.ok:
            log.warning("verification failed: %s", " | ".join(p[:200] for p in report.problems))
            if not report.build_ok and not packages:
                # nothing in the bundle influences the image -> healing cannot help
                raise PipelineError("environment image failed to build (see evidence/build.log)")
        return report

    def _converge(self, verifier: ConvergenceVerifier, env_builder: PythonEnvironmentBuilder,
                  engine: SynthesisEngine, ctx: ContextPackage, spec: BriefSpec, syn: Synthesis, *,
                  tree: Any, profile: Any) -> tuple[VerificationReport, int, Synthesis]:
        """verify -> (focused) heal -> verify ..., then prune what never converged."""
        cfg = self.config
        need_build = True
        packages = list(syn.extra_pip_packages)
        attempts = 0
        report: VerificationReport | None = None
        heal_rejection: str | None = None   # validator / guard verdict on the last healed bundle
        staged_notes: list[str] = []
        for attempt in range(cfg.limits.max_retries + 1):
            attempts = attempt + 1
            if heal_rejection is None:
                log.info("verification attempt %d/%d", attempts, cfg.limits.max_retries + 1)
                report = self._verify_once(verifier, syn.manifest, need_build=need_build, packages=packages)
                if report.ok:
                    break
                staged_notes = verifier.locate_failing_step(report, syn.solution, syn.manifest, syn.coverage)
                if staged_notes:
                    log.info("staged diagnosis: %s", staged_notes)
                    write_json(self.evidence_dir / "verification.json", report.to_dict())
                self._archive_attempt(attempts)
            else:
                # the previous healed bundle never reached the sandbox: re-verifying the old one is
                # pointless, ask the model again with the verdict attached
                log.info("healing attempt %d/%d (bundle rejected before the sandbox)", attempts, cfg.limits.max_retries + 1)
            assert report is not None
            if attempt == cfg.limits.max_retries:
                break

            attribution = attribute_problems(report.problems, syn.manifest, syn.coverage)
            frozen = attribution.converged([r.id for r in spec.requirements]) if report.build_ok else []
            focus = {rid: ps for rid, ps in attribution.by_requirement.items()}
            problems = list(report.problems) + staged_notes
            if heal_rejection:
                problems.append(f"your previous corrected bundle was rejected before execution: {heal_rejection}")
            log.info("healing: frozen=%s focus=%s", frozen, sorted(focus))
            try:
                healed = engine.heal(ctx, syn, problems, report.logs, max_chars=cfg.limits.max_context_chars,
                                     frozen=frozen, focus=focus)
            except SynthesisError as exc:
                log.warning("healing produced an invalid bundle: %s", exc)
                heal_rejection = str(exc)[:2000]
                continue
            heal_rejection = self._healing_guard(report, syn, healed, frozen)
            if heal_rejection:
                log.warning("healed bundle rejected: %s", heal_rejection[:200])
                continue
            syn = healed
            materialize(self.task_dir, syn)
            self._write_synthesis_evidence(syn, healed_attempt=attempts)
            need_build = (not report.build_ok) or syn.extra_pip_packages != packages
            packages = list(syn.extra_pip_packages)
            if need_build:
                env_builder.build(task_dir=self.task_dir, repo=cfg.repository, tree=tree, profile=profile,
                                  extra_packages=packages)

        assert report is not None
        if not report.ok and report.build_ok and cfg.limits.prune_unconverged:
            pruned = self._prune(engine, spec, syn, report)
            if pruned:
                attempts += 1
                log.info("verification after pruning %s", pruned)
                report = self._verify_once(verifier, syn.manifest, need_build=False, packages=packages)
                if not report.ok:
                    self._archive_attempt(attempts)
        return report, attempts, syn

    @staticmethod
    def _healing_guard(report: VerificationReport, previous: Synthesis, healed: Synthesis,
                       frozen: list[str]) -> str | None:
        """Reject a healed bundle that bends the reference solution instead of the tests, or that
        touches a requirement that already converged."""
        changed = previous.solution.changed_steps(healed.solution)
        touched_frozen = [r for r in changed if r in frozen]
        if touched_frozen:
            return (f"the solve step(s) of converged requirement(s) {', '.join(touched_frozen)} were modified. "
                    "Those requirements are frozen: return their scripts byte-for-byte unchanged and work "
                    "only on the requirements listed in <focus>.")
        for rid in frozen:
            before = {t: cat for cat in previous.manifest for t in previous.manifest[cat] if t in previous.tests_of(rid)}
            after = {t: cat for cat in healed.manifest for t in healed.manifest[cat]}
            moved = [t for t, cat in before.items() if after.get(t) != cat]
            if moved:
                return (f"tests of converged requirement {rid} were removed or re-categorised: {', '.join(moved)}. "
                        "Keep them exactly as they were.")
        if changed and not solution_change_justified(report):
            return ("the solve step(s) " + ", ".join(changed) + " were modified although every failure was on the "
                    "BASE run (pass_to_pass / anti_cheat failing on the original code, or fail_to_pass passing on "
                    "it). Those are TEST defects: keep the steps exactly as they were and fix the tests instead.")
        return None

    def _prune(self, engine: SynthesisEngine, spec: BriefSpec, syn: Synthesis,
               report: VerificationReport) -> list[str]:
        """Drop the requirements that never converged. The instruction is rewritten by the model (a
        failed rewrite fails the case; nothing is patched by hand)."""
        attribution = attribute_problems(report.problems, syn.manifest, syn.coverage)
        ids, why_not = prune_candidates(attribution, spec, syn)
        if not ids:
            log.warning("pruning not possible: %s", why_not)
            self.limitations.append(f"pruning not possible: {why_not}")
            return []
        pruned: dict[str, str] = {}
        for rid in ids:
            reason = ("did not converge after healing: " + " | ".join(p[:200] for p in attribution.by_requirement[rid][:3]))
            req = spec.prune(rid, reason)
            removed = syn.prune_requirement(rid, reason)
            pruned[rid] = reason
            title = req.title if req else rid
            self.limitations.append(f"requirement {rid} ({title}) was pruned: {reason}; removed tests: "
                                    + (", ".join(removed) or "none"))
            log.warning("pruned requirement %s (%s); removed tests %s", rid, title, removed)
        write_json(self.evidence_dir / "pruned.json", {"requirements": spec.pruned, "steps": syn.solution.pruned})
        write_json(self.evidence_dir / "brief_spec.json", spec.to_dict())
        syn.instruction_md = engine.rewrite_instruction(syn, pruned)
        materialize(self.task_dir, syn)
        self._write_synthesis_evidence(syn, pruned_requirements=list(pruned))
        return list(pruned)
