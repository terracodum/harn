"""Isolated docker build / run with `--network none` (Stage 5.1-5.3)."""
from __future__ import annotations

import logging
import shutil
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger(__name__)


class DockerError(RuntimeError):
    pass


@dataclass
class ProcResult:
    exit_code: int | None
    duration_sec: float
    timed_out: bool
    log_path: str
    command: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Mount:
    host: Path
    container: str
    read_only: bool = True

    def as_arg(self) -> str:
        suffix = ":ro" if self.read_only else ""
        return f"{self.host.resolve()}:{self.container}{suffix}"


class DockerRunner:
    def __init__(self, docker_bin: str = "docker") -> None:
        self.docker = docker_bin

    def available(self) -> tuple[bool, str]:
        if shutil.which(self.docker) is None:
            return False, f"{self.docker} not found in PATH"
        try:
            proc = subprocess.run([self.docker, "info", "--format", "{{.ServerVersion}}"],
                                  capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, str(exc)
        if proc.returncode != 0:
            return False, (proc.stderr or proc.stdout).strip()[:500]
        return True, proc.stdout.strip()

    def _execute(self, cmd: list[str], log_path: Path, timeout: int, *, container_name: str | None = None) -> ProcResult:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        timed_out = False
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write("$ " + " ".join(cmd) + "\n\n")
            fh.flush()
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                if container_name:
                    subprocess.run([self.docker, "kill", container_name], capture_output=True)
                proc.kill()
                proc.wait()
                fh.write(f"\n[harness] timed out after {timeout}s\n")
        return ProcResult(proc.returncode, round(time.monotonic() - started, 3), timed_out, str(log_path),
                          command=" ".join(cmd))

    def build(self, context_dir: Path, tag: str, *, log_path: Path, timeout: int) -> ProcResult:
        cmd = [self.docker, "build", "--progress=plain", "-t", tag, str(context_dir.resolve())]
        log.info("docker build %s", tag)
        return self._execute(cmd, log_path, timeout)

    def run(self, tag: str, *, mounts: list[Mount], command: str, log_path: Path, timeout: int,
            network_none: bool = True, cpus: int | None = None, memory_mb: int | None = None) -> ProcResult:
        name = f"harness-{uuid.uuid4().hex[:12]}"
        cmd = [self.docker, "run", "--rm", "--name", name]
        if network_none:
            cmd += ["--network", "none"]
        if cpus:
            cmd += ["--cpus", str(cpus)]
        if memory_mb:
            cmd += ["--memory", f"{memory_mb}m"]
        for m in mounts:
            cmd += ["-v", m.as_arg()]
        cmd += [tag, "sh", "-c", command]
        log.info("docker run %s: %s", tag, command)
        return self._execute(cmd, log_path, timeout, container_name=name)

    def remove_image(self, tag: str) -> None:
        subprocess.run([self.docker, "rmi", "-f", tag], capture_output=True)
