"""Reference solution as an ordered chain of per-requirement STRUCTURED edits.

    task/solution/
    ├── solve.sh        # self-contained: every step inlined, prints "[solve] step R<i>" before each
    ├── solve_R1.sh     # the same step R1 alone (extra file, used for staged diagnostics)
    └── solve_R3.sh     # step R3 alone (R2 was an invariant: no step)

A step is a list of edits (`replace` / `create` / `delete`), not free-form shell. That is what
lets the harness (a) compute the repository state after steps R1..Ri-1 in memory and show it to
the model when it works on Ri, (b) reject a bundle BEFORE the sandbox when an `old` snippet does
not occur exactly once in that state, (c) know which later steps break when a step is pruned,
and (d) render solve.sh deterministically, with no shell-quoting mistakes from the model.
"""
from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from harness.providers.python.env_builder import write_lf

_RID = re.compile(r"^R(\d+)$")
STEP_MARKER = "[solve] step "          # printed by solve.sh before each step (see solve.log)
EDIT_OPS = ("replace", "create", "delete")

ReadFile = Callable[[str], "str | None"]    # relative POSIX path -> text, None when absent


def requirement_order(rid: str) -> int:
    m = _RID.match(rid)
    return int(m.group(1)) if m else 10**6


def step_file(rid: str) -> str:
    return f"solve_{rid}.sh"


def _safe_rel_path(path: str) -> bool:
    if not path or path.startswith(("/", "\\")) or "\\" in path or ":" in path:
        return False
    norm = posixpath.normpath(path)
    return norm == path.rstrip("/") and not norm.startswith("..") and "/../" not in f"/{norm}/" \
        and not norm.startswith("tests/") and norm != "tests"


@dataclass(frozen=True)
class Edit:
    op: str
    path: str
    old: str = ""
    new: str = ""
    content: str = ""

    def to_dict(self) -> dict[str, str]:
        d = {"op": self.op, "path": self.path}
        if self.op == "replace":
            d["old"], d["new"] = self.old, self.new
        elif self.op == "create":
            d["content"] = self.content
        return d


def parse_edits(items: Any, where: str) -> tuple[list[Edit], list[str]]:
    problems: list[str] = []
    edits: list[Edit] = []
    if not isinstance(items, list) or not items:
        return [], [f"{where}: edits must be a non-empty list"]
    for n, item in enumerate(items, 1):
        tag = f"{where} edit #{n}"
        if not isinstance(item, dict):
            problems.append(f"{tag}: not an object")
            continue
        op = str(item.get("op", "")).strip()
        path = str(item.get("path", "")).strip()
        if op not in EDIT_OPS:
            problems.append(f"{tag}: unknown op {op!r} (expected {', '.join(EDIT_OPS)})")
            continue
        if not _safe_rel_path(path):
            problems.append(f"{tag}: path {path!r} must be a normalised relative path inside the repository (not under tests/)")
            continue
        if op == "replace":
            old, new = item.get("old"), item.get("new")
            if not isinstance(old, str) or not old:
                problems.append(f"{tag}: replace needs a non-empty string `old`")
                continue
            if not isinstance(new, str):
                problems.append(f"{tag}: replace needs a string `new`")
                continue
            if old == new:
                problems.append(f"{tag}: `old` and `new` are identical")
                continue
            edits.append(Edit("replace", path, old=old, new=new))
        elif op == "create":
            content = item.get("content")
            if not isinstance(content, str):
                problems.append(f"{tag}: create needs a string `content`")
                continue
            edits.append(Edit("create", path, content=content))
        else:
            edits.append(Edit("delete", path))
    return edits, problems


def apply_edits(edits: list[Edit], state: dict[str, "str | None"], read_file: ReadFile, where: str) -> list[str]:
    """Apply `edits` to `state` in place (path -> text, None = deleted). Files not yet in `state`
    are read lazily through `read_file`. Returns the problems found; the state is only modified
    by edits that applied cleanly."""
    problems: list[str] = []
    for n, e in enumerate(edits, 1):
        tag = f"{where} edit #{n} ({e.op} {e.path})"
        if e.path not in state:
            state[e.path] = read_file(e.path)
        current = state[e.path]
        if e.op == "replace":
            if current is None:
                problems.append(f"{tag}: file does not exist at this point of the chain")
                continue
            count = current.count(e.old)
            if count != 1:
                hint = "not found" if count == 0 else f"found {count} times"
                problems.append(f"{tag}: `old` snippet {hint} in the file as left by the previous steps; "
                                f"copy the exact current text (first 60 chars of `old`: {e.old[:60]!r})")
                continue
            state[e.path] = current.replace(e.old, e.new, 1)
        elif e.op == "create":
            if current is not None:
                problems.append(f"{tag}: file already exists; use replace")
                continue
            state[e.path] = e.content
        else:
            if current is None:
                problems.append(f"{tag}: file does not exist")
                continue
            state[e.path] = None
    return problems


@dataclass
class StagedSolution:
    steps: dict[str, list[Edit]] = field(default_factory=dict)   # requirement id -> edits
    pruned: dict[str, str] = field(default_factory=dict)         # requirement id -> reason

    # ---------------------------------------------------------------- building
    @classmethod
    def from_items(cls, items: Iterable[Any]) -> tuple["StagedSolution", list[str]]:
        """Parse `solve_steps` from a model bundle. Returns (solution, problems)."""
        problems: list[str] = []
        steps: dict[str, list[Edit]] = {}
        for i, item in enumerate(items, 1):
            if not isinstance(item, dict):
                problems.append(f"solve_steps[{i}] is not an object")
                continue
            rid = str(item.get("requirement_id", "")).strip()
            if not _RID.match(rid):
                problems.append(f"solve_steps[{i}]: requirement_id {rid!r} must look like R1, R2, ...")
                continue
            if rid in steps:
                problems.append(f"solve_steps: requirement {rid} has two steps")
                continue
            edits, ep = parse_edits(item.get("edits"), f"solve_steps {rid}")
            problems += ep
            if edits and not ep:
                steps[rid] = edits
        ordered = dict(sorted(steps.items(), key=lambda kv: requirement_order(kv[0])))
        return cls(steps=ordered), problems

    @property
    def order(self) -> list[str]:
        return sorted(self.steps, key=requirement_order)

    def prefix(self, upto: str) -> "StagedSolution":
        """Steps R1..upto (inclusive) - used by the staged diagnostic runs."""
        keep = [r for r in self.order if requirement_order(r) <= requirement_order(upto)]
        return StagedSolution(steps={r: self.steps[r] for r in keep})

    def prune(self, rid: str, reason: str) -> bool:
        """Drop the step of `rid`; returns False when there was nothing to drop."""
        self.pruned[rid] = reason
        return self.steps.pop(rid, None) is not None

    def changed_steps(self, other: "StagedSolution") -> list[str]:
        """Requirement ids whose edits differ between self and `other` (added/removed count too)."""
        ids = set(self.steps) | set(other.steps)
        return sorted((r for r in ids if self.steps.get(r) != other.steps.get(r)), key=requirement_order)

    # ------------------------------------------------------------ application
    def state_after(self, read_file: ReadFile, upto: str | None = None, *,
                    steps: list[str] | None = None) -> tuple[dict[str, "str | None"], list[str]]:
        """Repository state (touched files only) after applying the given steps in chain order.
        `upto` = last step included (None = all); `steps` overrides the selection explicitly."""
        chosen = steps if steps is not None else \
            [r for r in self.order if upto is None or requirement_order(r) <= requirement_order(upto)]
        state: dict[str, str | None] = {}
        problems: list[str] = []
        for rid in sorted(chosen, key=requirement_order):
            problems += apply_edits(self.steps[rid], state, read_file, f"solve_steps {rid}")
        return state, problems

    def validate(self, read_file: ReadFile) -> list[str]:
        """Every step must apply cleanly on top of the previous ones."""
        return self.state_after(read_file)[1]

    def dependents(self, rid: str, read_file: ReadFile) -> list[str]:
        """Later steps that no longer apply once `rid` is removed."""
        rest = [r for r in self.order if r != rid]
        _, problems = self.state_after(read_file, steps=rest)
        broken = {m.group(1) for p in problems for m in [re.match(r"solve_steps (R\d+)", p)] if m}
        return sorted(broken - {rid}, key=requirement_order)

    def touched_paths(self, upto: str | None = None) -> list[str]:
        seen: dict[str, None] = {}
        for rid in self.order:
            if upto is not None and requirement_order(rid) > requirement_order(upto):
                break
            for e in self.steps[rid]:
                seen.setdefault(e.path, None)
        return list(seen)

    # --------------------------------------------------------------- rendering
    def _python_block(self, rid: str) -> str:
        lines = ["python - <<'HARNESS_STEP'", "import os", "from pathlib import Path", "",
                 f"STEP = {rid!r}", "EDITS = ["]
        for e in self.steps[rid]:
            lines.append("    {")
            for k, v in e.to_dict().items():
                lines.append(f"        {k!r}: {v!r},")
            lines.append("    },")
        lines += ["]", "",
                  "root = Path(os.environ.get('REPO_PATH', '.'))",
                  "for n, e in enumerate(EDITS, 1):",
                  "    p = root / e['path']",
                  "    if e['op'] == 'replace':",
                  "        s = p.read_text(encoding='utf-8')",
                  "        if s.count(e['old']) != 1:",
                  "            raise SystemExit(f'{STEP} edit {n}: expected exactly one occurrence in {p}')",
                  "        p.write_text(s.replace(e['old'], e['new'], 1), encoding='utf-8')",
                  "    elif e['op'] == 'create':",
                  "        if p.exists():",
                  "            raise SystemExit(f'{STEP} edit {n}: {p} already exists')",
                  "        p.parent.mkdir(parents=True, exist_ok=True)",
                  "        p.write_text(e['content'], encoding='utf-8')",
                  "    elif e['op'] == 'delete':",
                  "        p.unlink()",
                  "HARNESS_STEP"]
        return "\n".join(lines)

    def render_step(self, rid: str) -> str:
        return "\n".join(["#!/bin/sh", f"# Step {rid} of the reference solution (generated by benchmark-harness).",
                          "set -eu", f'echo "{STEP_MARKER}{rid}"', self._python_block(rid)]) + "\n"

    def orchestrator(self) -> str:
        """solve.sh: self-contained, every step inlined in chain order."""
        lines = ["#!/bin/sh", "# Reference solution generated by benchmark-harness: one step per requirement, in order.",
                 "set -eu"]
        for rid in self.order:
            lines += [f'echo "{STEP_MARKER}{rid}"', self._python_block(rid)]
        lines.append('echo "[solve] done"')
        return "\n".join(lines) + "\n"

    def combined(self) -> str:
        """Text of solve.sh (prompts, evidence, change detection)."""
        return self.orchestrator()

    def write(self, solution_dir: Path) -> None:
        solution_dir.mkdir(parents=True, exist_ok=True)
        for old in solution_dir.glob("solve*.sh"):
            old.unlink()
        for rid in self.order:
            write_lf(solution_dir / step_file(rid), self.render_step(rid))
        write_lf(solution_dir / "solve.sh", self.orchestrator())

    def to_items(self) -> list[dict[str, Any]]:
        return [{"requirement_id": rid, "edits": [e.to_dict() for e in self.steps[rid]]} for rid in self.order]


def failing_step_from_log(solve_log: str) -> str | None:
    """Requirement id of the step that was running when solve.sh stopped (last marker in the log)."""
    last: str | None = None
    for line in solve_log.splitlines():
        line = line.strip()
        if line.startswith(STEP_MARKER):
            last = line[len(STEP_MARKER):].strip()
        elif line == "[solve] done":
            return None
    return last
