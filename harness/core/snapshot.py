"""Deterministic SHA-256 of a repository tree (Stage 6.2).

Algorithm (from HARNESS_ARCHITECTURE.md):
  * take every regular file, sort by POSIX path (lexicographic, byte order);
  * for each: path + "\0" + sha256_hex(content) + "\n";
  * sha256 of the resulting UTF-8 byte stream.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from harness.localization.code_retriever import DEFAULT_IGNORED_DIRS


def iter_regular_files(root: Path, ignored_dirs: Iterable[str] = DEFAULT_IGNORED_DIRS) -> list[Path]:
    ignored = set(ignored_dirs)
    result: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(root)
        if any(part in ignored for part in rel.parts[:-1]):
            continue
        result.append(path)
    return result


def snapshot_sha256(root: Path, *, ignore_dirs: Iterable[str] = (".git",)) -> str:
    """Hash the repository. Only `.git` is ignored by default: the snapshot must
    reflect what actually sits on disk, not the filtered project tree."""
    root = Path(root)
    files = iter_regular_files(root, ignored_dirs=ignore_dirs)
    entries = sorted(
        (path.relative_to(root).as_posix(), path) for path in files
    )
    stream = hashlib.sha256()
    for posix_path, path in sorted(entries, key=lambda e: e[0].encode("utf-8")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        stream.update(f"{posix_path}\0{digest}\n".encode("utf-8"))
    return stream.hexdigest()
