import hashlib

from harness.core.snapshot import snapshot_sha256


def _manual(root):
    files = sorted(p for p in root.rglob("*") if p.is_file() and ".git" not in p.relative_to(root).parts)
    stream = hashlib.sha256()
    for p in sorted(files, key=lambda p: p.relative_to(root).as_posix().encode()):
        stream.update(f"{p.relative_to(root).as_posix()}\0{hashlib.sha256(p.read_bytes()).hexdigest()}\n".encode())
    return stream.hexdigest()


def test_matches_spec_algorithm(tmp_path):
    (tmp_path / "b.txt").write_bytes(b"bbb")
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "z.py").write_bytes(b"print(1)\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_bytes(b"ref")
    assert snapshot_sha256(tmp_path) == _manual(tmp_path)


def test_deterministic_and_content_sensitive(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    h1 = snapshot_sha256(tmp_path)
    assert snapshot_sha256(tmp_path) == h1
    (tmp_path / "a.py").write_text("x = 2\n")
    assert snapshot_sha256(tmp_path) != h1
