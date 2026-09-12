from harness.providers.base import RunResult, TestStatus
from harness.providers.python.test_runner import check_run
from harness.providers.python.verifier_template import compute_reward, parse_junit, to_container_nodeid

XML = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" errors="1" failures="1" skipped="1" tests="5">
<testcase classname="tests.test_a" name="test_ok" time="0.001"/>
<testcase classname="tests.test_a" name="test_bad" time="0.001"><failure message="assert 1 == 2">Trace</failure></testcase>
<testcase classname="tests.test_a.TestGroup" name="test_in_class" time="0.001"/>
<testcase classname="tests.test_a" name="test_param[1-2]" time="0.001"><skipped message="xfail">x</skipped></testcase>
<testcase classname="tests.sub.test_b" name="test_err" time="0.001"><error message="SyntaxError: bad">Trace</error></testcase>
</testsuite></testsuites>
"""


def test_parse_junit_maps_ids(tmp_path):
    xml = tmp_path / "tests.xml"
    xml.write_text(XML, encoding="utf-8")
    res = parse_junit(xml, {"tests/test_a.py", "tests/sub/test_b.py"})
    assert res["tests/test_a.py::test_ok"]["outcome"] == "passed"
    assert res["tests/test_a.py::test_bad"]["outcome"] == "failed"
    assert res["tests/test_a.py::TestGroup::test_in_class"]["outcome"] == "passed"
    assert res["tests/test_a.py::test_param[1-2]"]["outcome"] == "skipped"
    assert res["tests/sub/test_b.py::test_err"]["outcome"] == "error"


def test_compute_reward_requires_all_passed(tmp_path):
    xml = tmp_path / "tests.xml"
    xml.write_text(XML, encoding="utf-8")
    res = parse_junit(xml, {"tests/test_a.py", "tests/sub/test_b.py"})
    assert compute_reward(["tests/test_a.py::test_ok", "tests/test_a.py::TestGroup::test_in_class"], res)[0] == 1
    assert compute_reward(["tests/test_a.py::test_ok", "tests/test_a.py::test_param[1-2]"], res)[0] == 0
    assert compute_reward(["tests/test_a.py::missing"], res)[0] == 0
    assert compute_reward([], res)[0] == 0


def test_to_container_nodeid():
    assert to_container_nodeid("tests/test_a.py::test_x", "/tests") == "/tests/test_a.py::test_x"


MANIFEST = {"fail_to_pass": ["tests/t.py::f2p"], "pass_to_pass": ["tests/t.py::p2p"], "anti_cheat": ["tests/t.py::ac"]}


def _run(reward, **outcomes):
    tests = {tid: TestStatus(tid, out, "AssertionError" if out == "failed" else "") for tid, out in outcomes.items()}
    return RunResult(reward=reward, tests=tests, exit_code=1, duration_sec=1.0)


def test_check_base_ok():
    rep = check_run("base", _run(0, **{"tests/t.py::f2p": "failed", "tests/t.py::p2p": "passed", "tests/t.py::ac": "passed"}), MANIFEST)
    assert rep.ok, rep.problems


def test_check_base_detects_leaky_f2p():
    rep = check_run("base", _run(1, **{"tests/t.py::f2p": "passed", "tests/t.py::p2p": "passed", "tests/t.py::ac": "passed"}), MANIFEST)
    assert not rep.ok and any("passed on original" in p for p in rep.problems)


def test_check_base_flags_syntax_error():
    res = _run(0, **{"tests/t.py::f2p": "error", "tests/t.py::p2p": "passed", "tests/t.py::ac": "passed"})
    res.tests["tests/t.py::f2p"].message = "SyntaxError: invalid syntax"
    rep = check_run("base", res, MANIFEST)
    assert not rep.ok and any("environment/syntax" in p for p in rep.problems)


def test_check_oracle_requires_all_green():
    rep = check_run("oracle", _run(1, **{"tests/t.py::f2p": "passed", "tests/t.py::p2p": "passed", "tests/t.py::ac": "passed"}), MANIFEST)
    assert rep.ok
    rep = check_run("oracle", _run(0, **{"tests/t.py::f2p": "failed", "tests/t.py::p2p": "passed", "tests/t.py::ac": "passed"}), MANIFEST)
    assert not rep.ok
