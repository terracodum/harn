"""Orchestrator: one run = discovery + brief analysis, then one benchmark case per bug/feature/change
requirement. Every case is built on the ORIGINAL repository, verified on its own and written as a
complete PROTOCOL.md case (task/, evidence/, result.json) under output_dir/cases/<R>/.

Failure policy: every LLM step either succeeds or raises (`LLMError`, `SynthesisError`,
`PipelineError`). A case whose bundle cannot be produced or never converges is reported as
`case_failed` with the reason - it is always listed, never dropped. Nothing here catches a model
failure to continue with a heuristic result; the `except` blocks only turn exceptions into
result.json so that a run always leaves a readable trace.
"""
from __future__ import annotations

import logging
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from harness.core.artifacts import write_json, write_result, write_run_result, write_task_toml
from harness.core.brief import BriefSpec, Requirement, analyze_brief, case_requirements
from harness.core.config import CaseConfig
from harness.core.errors import PipelineError
from harness.core.llm.base_client import BaseLLMClient, LLMError
from harness.core.snapshot import snapshot_sha256
from harness.core.synthesis import Synthesis, SynthesisEngine, SynthesisError, materialize
from harness.localization.code_retriever import ContextPackage, build_project_tree, localize, read_text
from harness.providers.python.detector import PythonStackDetector
from harness.providers.python.env_builder import PythonEnvironmentBuilder
from harness.providers.python.test_runner import PytestJUnitRunner
from harness.validation.convergence import ConvergenceVerifier, VerificationReport, solution_change_justified
from harness.validation.docker_runner import DockerRunner

log = logging.getLogger("harness")

__all__ = ["Pipeline", "PipelineError", "CaseOutcome"]

CASE_READY = "ready"
CASE_FAILED = "case_failed"


@dataclass
class CaseOutcome:
    requirement_id: str
    title: str
    case_id: str
    path: str                       # relative to output_dir, e.g. cases/R1
    status: str = CASE_FAILED       # ready | case_failed
    attempts: int = 0
    error: str | None = None
    failed_stage: str | None = None
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["task_path"] = f"{self.path}/task"
        d["evidence_path"] = f"{self.path}/evidence"
        d["result_path"] = f"{self.path}/result.json"
        return d


def case_id_for(base_case_id: str, rid: str) -> str:
    return f"{base_case_id}-{rid.lower()}"


def image_tag_for(case_id: str, seed: int) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", case_id.lower()).strip("-.")
    return f"harness-{slug}:{seed}"


class _Timer:
    def __init__(self, times: dict[str, float], name: str, owner: Any) -> None:
        self.times, self.name, self.owner = times, name, owner

    def __enter__(self):
        log.info("=== %s", self.name)
        self.owner.current_stage = self.name
        self.t = time.monotonic()
        return self

    def __exit__(self, *exc):
        self.times[self.name] = round(time.monotonic() - self.t, 3)
        return False


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
        self.evidence_dir = self.output_dir / "evidence"
        self.cases_dir = self.output_dir / "cases"
        self.work_dir = self.output_dir / ".work"
        self.limitations: list[str] = []
        self.stage_times: dict[str, float] = {}
        self.current_stage = "init"
        self.cases: list[CaseOutcome] = []
        self.docker_version = ""

    def _stage(self, name: str) -> _Timer:
        return _Timer(self.stage_times, name, self)

    # -------------------------------------------------------------------- run
    def run(self) -> dict[str, Any]:
        cfg = self.config
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(exist_ok=True)
        status, error = "failed", None
        snapshot_before = ""
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
                for note in profile.notes:
                    self.limitations.append(f"stack: {note}")

            with self._stage("2a-brief-analysis"):
                spec = analyze_brief(self.llm, cfg.brief, [f.rel_path for f in tree.trusted_files],
                                     language_name=cfg.language_name)
                write_json(self.evidence_dir / "brief_spec.json", spec.to_dict())
                log.info("requirements: %s", [f"{r.id}[{r.kind}]:{r.title[:60]}" for r in spec.requirements])
                for a in spec.assumptions:
                    self.limitations.append(f"assumption: {a}")
                for q in spec.ambiguities:
                    self.limitations.append(f"ambiguity: {q}")
                targets = case_requirements(spec)
                if not targets:
                    raise PipelineError("the brief yields no bug/feature/change requirement: nothing to build a case from "
                                        f"(requirements: {[f'{r.id}:{r.kind}' for r in spec.requirements]})")
                skipped = [r for r in spec.requirements if r not in targets and r.kind in ("bug", "feature", "change")]
                for r in skipped:
                    self.limitations.append(f"requirement {r.id} ({r.title}) is marked not testable: no case built for it")
                log.info("cases to build: %s", [r.id for r in targets])

            if not self.skip_docker:
                ok, info = self.docker.available()
                if not ok:
                    raise PipelineError(f"docker is not available: {info}")
                self.docker_version = info

            read_file = lambda rel: read_text(cfg.repository / rel)  # noqa: E731
            for r in targets:
                with self._stage(f"case-{r.id}"):
                    outcome = self._build_case(r, spec, tree=tree, profile=profile, read_file=read_file,
                                               snapshot=snapshot_before)
                self.cases.append(outcome)
                log.info("case %s: %s%s", r.id, outcome.status, f" ({outcome.error})" if outcome.error else "")

            with self._stage("6-packaging"):
                if snapshot_sha256(cfg.repository) != snapshot_before:
                    raise PipelineError("source repository changed during generation")
                failed = [c for c in self.cases if c.status != CASE_READY]
                broken = [c for c in failed if c.error]
                if broken:
                    error = (f"{len(broken)} of {len(self.cases)} case(s) failed: "
                             + ", ".join(f"{c.requirement_id} ({c.error})"[:200] for c in broken))
                elif not failed:
                    status = "ready"
                # else: cases packaged but not verified (--skip-docker) - listed in limitations, no error
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
            write_json(self.evidence_dir / "summary.json", {
                "status": status, "stage_times_sec": self.stage_times,
                "cases": [c.to_dict() for c in self.cases],
            })
            shutil.rmtree(self.work_dir, ignore_errors=True)
            result = write_run_result(
                self.output_dir, config=cfg, status=status, error=error, limitations=self.limitations,
                snapshot_sha256=snapshot_before, cases=[c.to_dict() for c in self.cases],
                extra={"failed_stage": None if status != "failed" else self.current_stage,
                       "llm": {k: v for k, v in self.llm.tracker.summary().items() if k != "calls"}
                       | {"details": "evidence/llm_usage.json"}})
        return result

    # ------------------------------------------------------------------- case
    def _build_case(self, req: Requirement, spec: BriefSpec, *, tree: Any, profile: Any, read_file: Any,
                    snapshot: str) -> CaseOutcome:
        cfg = self.config
        case_id = case_id_for(cfg.case_id, req.id)
        case_dir = self.cases_dir / req.id
        task_dir, evidence_dir = case_dir / "task", case_dir / "evidence"
        task_dir.mkdir(parents=True, exist_ok=True)
        evidence_dir.mkdir(parents=True, exist_ok=True)
        outcome = CaseOutcome(requirement_id=req.id, title=req.title, case_id=case_id,
                              path=case_dir.relative_to(self.output_dir).as_posix())
        cspec = spec.case_spec(req.id)
        write_json(evidence_dir / "case_spec.json", cspec.to_dict())
        others = [f"{r.id}: {r.title}" for r in case_requirements(spec) if r.id != req.id]
        engine: SynthesisEngine | None = None
        report: VerificationReport | None = None
        manifest: dict[str, list[str]] = {"fail_to_pass": [], "pass_to_pass": [], "anti_cheat": []}
        calls_before = len(self.llm.tracker.calls)
        stage = "localization"
        try:
            ctx: ContextPackage = localize(
                tree, cfg.brief, spec=cspec, index_dir=self.work_dir / "index" / req.id, embedder=self.embedder,
                backend=self.search_backend, max_files=cfg.limits.max_context_files, max_chars=cfg.limits.max_context_chars)
            write_json(evidence_dir / "localization.json", ctx.summary())
            log.info("[%s] context files: %s", req.id, [f.path for f in ctx.files])

            stage = "synthesis"
            engine = SynthesisEngine(self.llm, brief=cfg.brief, profile=profile, spec=cspec, read_file=read_file,
                                     instruction_language=cfg.language_name, difficulty=cfg.difficulty,
                                     other_cases=others)
            env_builder = PythonEnvironmentBuilder()
            syn: Synthesis = engine.synthesize(ctx, max_chars=cfg.limits.max_context_chars)
            materialize(task_dir, syn)
            manifest = syn.manifest
            self._write_synthesis_evidence(evidence_dir, syn)

            stage = "environment"
            env_builder.build(task_dir=task_dir, repo=cfg.repository, tree=tree, profile=profile,
                              extra_packages=syn.extra_pip_packages)

            if self.skip_docker:
                outcome.limitations.append("verification skipped (--skip-docker): Base/Oracle runs not executed, "
                                           "the case is packaged but NOT verified")
            else:
                stage = "convergence"
                image_tag = image_tag_for(case_id, cfg.seed)
                verifier = ConvergenceVerifier(self.docker, PytestJUnitRunner(), task_dir=task_dir,
                                               evidence_dir=evidence_dir, limits=cfg.limits, image_tag=image_tag,
                                               isolated_runs=self.isolated_runs, docker_version=self.docker_version)
                try:
                    report, attempts, syn = self._converge(verifier, env_builder, engine, ctx, syn, evidence_dir,
                                                           task_dir=task_dir, tree=tree, profile=profile)
                finally:
                    if not self.keep_image:
                        self.docker.remove_image(image_tag)
                manifest = syn.manifest
                outcome.attempts = attempts
                if report.ok:
                    outcome.status = CASE_READY
                else:
                    outcome.error = "convergence failed: " + " | ".join(p[:300] for p in report.problems[:10])

            stage = "packaging"
            write_task_toml(task_dir, cfg, manifest, name=case_id, description=req.title or cspec.summary,
                            bank_domain=spec.bank_domain)
            if profile.uses_postgres:
                outcome.limitations.append("PostgreSQL provisioned with a default cluster; only `alembic upgrade head` is applied")
        except (SynthesisError, LLMError, PipelineError) as exc:
            kind = {"SynthesisError": "bundle rejected by the validator before any sandbox run",
                    "LLMError": "LLM step failed"}.get(type(exc).__name__, "case failed")
            outcome.error = f"{kind} ({stage}): {exc}"
            outcome.failed_stage = stage
            log.error("[%s] %s", req.id, outcome.error)
        finally:
            if outcome.status != CASE_READY:
                outcome.failed_stage = outcome.failed_stage or stage
            self.llm.tracker.write(evidence_dir / "llm_usage.json", start=calls_before)
            if engine is not None:
                for i, item in enumerate(engine.raw_history, 1):
                    write_json(evidence_dir / "llm_responses" / f"{i:02d}_{item['purpose']}.json", item["data"])
            write_json(evidence_dir / "summary.json", {
                "status": outcome.status, "attempts": outcome.attempts,
                "runs": report.runs if report else [],
                "isolated": report.isolated if report else {},
                "problems": report.problems if report else ([outcome.error] if outcome.error else []),
            })
            write_result(case_dir, config=cfg, case_id=case_id, status="ready" if outcome.status == CASE_READY else "failed",
                         error=outcome.error, limitations=outcome.limitations, attempts=outcome.attempts,
                         snapshot_sha256=snapshot,
                         extra={"case_status": outcome.status, "requirement_id": req.id, "requirement": req.title,
                                "failed_stage": outcome.failed_stage,
                                "llm": {k: v for k, v in self.llm.tracker.summary(start=calls_before).items() if k != "calls"}
                                | {"details": "evidence/llm_usage.json"}})
        return outcome

    @staticmethod
    def _write_synthesis_evidence(evidence_dir: Path, syn: Synthesis, **extra: Any) -> None:
        write_json(evidence_dir / "synthesis.json", {
            "root_cause": syn.root_cause, "notes": syn.notes, "manifest": syn.manifest, "coverage": syn.coverage,
            "edits": syn.edits, "extra_pip_packages": syn.extra_pip_packages, **extra})

    @staticmethod
    def _archive_attempt(evidence_dir: Path, task_dir: Path, n: int) -> None:
        dest = evidence_dir / "attempts" / f"{n:02d}"
        dest.mkdir(parents=True, exist_ok=True)
        for item in ("base", "oracle", "isolated", "build.log"):
            src = evidence_dir / item
            if src.exists():
                shutil.move(str(src), str(dest / item))
        for item in ("tests", "solution", "instruction.md"):
            src = task_dir / item
            if src.exists():
                if src.is_dir():
                    shutil.copytree(src, dest / "task" / item)
                else:
                    (dest / "task").mkdir(exist_ok=True)
                    shutil.copy2(src, dest / "task" / item)

    # ------------------------------------------------------------ convergence
    def _verify_once(self, verifier: ConvergenceVerifier, manifest: dict[str, list[str]], evidence_dir: Path, *,
                     need_build: bool, packages: list[str]) -> VerificationReport:
        report = verifier.verify(manifest, need_build=need_build)
        if need_build and not report.build_ok:
            # apt/pip mirrors flake: one immediate rebuild before judging the bundle
            log.warning("docker build failed, retrying the build once")
            shutil.copy2(evidence_dir / "build.log", evidence_dir / "build.first-try.log")
            report = verifier.verify(manifest, need_build=True)
        write_json(evidence_dir / "verification.json", report.to_dict())
        if not report.ok:
            log.warning("verification failed: %s", " | ".join(p[:200] for p in report.problems))
            if not report.build_ok and not packages:
                # nothing in the bundle influences the image -> healing cannot help
                raise PipelineError("environment image failed to build (see evidence/build.log)")
        return report

    def _converge(self, verifier: ConvergenceVerifier, env_builder: PythonEnvironmentBuilder,
                  engine: SynthesisEngine, ctx: ContextPackage, syn: Synthesis, evidence_dir: Path, *,
                  task_dir: Path, tree: Any, profile: Any) -> tuple[VerificationReport, int, Synthesis]:
        """verify -> heal -> verify ... up to limits.max_retries healing rounds."""
        cfg = self.config
        need_build = True
        packages = list(syn.extra_pip_packages)
        attempts = 0
        report: VerificationReport | None = None
        heal_rejection: str | None = None   # validator / guard verdict on the last healed bundle
        for attempt in range(cfg.limits.max_retries + 1):
            attempts = attempt + 1
            if heal_rejection is None:
                log.info("verification attempt %d/%d", attempts, cfg.limits.max_retries + 1)
                report = self._verify_once(verifier, syn.manifest, evidence_dir, need_build=need_build, packages=packages)
                if report.ok:
                    break
                self._archive_attempt(evidence_dir, task_dir, attempts)
            else:
                # the previous healed bundle never reached the sandbox: re-verifying the old one is
                # pointless, ask the model again with the verdict attached
                log.info("healing attempt %d/%d (bundle rejected before the sandbox)", attempts, cfg.limits.max_retries + 1)
            assert report is not None
            if attempt == cfg.limits.max_retries:
                break
            problems = list(report.problems)
            if heal_rejection:
                problems.append(f"your previous corrected bundle was rejected before execution: {heal_rejection}")
            try:
                healed = engine.heal(ctx, syn, problems, report.logs, max_chars=cfg.limits.max_context_chars)
            except SynthesisError as exc:
                log.warning("healing produced an invalid bundle: %s", exc)
                heal_rejection = str(exc)[:2000]
                continue
            if healed.edits != syn.edits and not solution_change_justified(report):
                # guard against "fixing" a wrong test by bending the reference solution
                heal_rejection = ("the edits were modified although every failure was on the BASE run "
                                  "(pass_to_pass / anti_cheat failing on the original code, or fail_to_pass passing "
                                  "on it). Those are TEST defects: keep the edits exactly as they were and fix the "
                                  "tests / categorisation instead.")
                log.warning("healed bundle rejected: unjustified change of the edits")
                continue
            syn = healed
            heal_rejection = None
            materialize(task_dir, syn)
            self._write_synthesis_evidence(evidence_dir, syn, healed_attempt=attempts)
            need_build = (not report.build_ok) or syn.extra_pip_packages != packages
            packages = list(syn.extra_pip_packages)
            if need_build:
                env_builder.build(task_dir=task_dir, repo=cfg.repository, tree=tree, profile=profile,
                                  extra_packages=packages)
        assert report is not None
        return report, attempts, syn
