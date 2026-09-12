import tomllib

from harness.core.artifacts import toml_dumps


def test_toml_roundtrip():
    doc = {
        "schema_version": "1.1", "case_id": "x", "seed": 3, "authors": ["a b"],
        "fail_to_pass": ["tests/t.py::a", "tests/t.py::b[1-2]"], "pass_to_pass": [],
        "limits": {"build_timeout_sec": 10, "flag": True},
        "environment": {"test_command": "sh /tests/test.sh", "note": 'quote " inside'},
    }
    assert tomllib.loads(toml_dumps(doc)) == doc
