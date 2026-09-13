from harness.providers.python.test_runner import CheckReport
from harness.validation.convergence import VerificationReport, attribute_problems, solution_change_justified


def _report(base: dict[str, str], oracle: dict[str, str], problems: list[str]) -> VerificationReport:
    return VerificationReport(ok=False, build_ok=True, problems=problems,
                              base=CheckReport("base", False, table=base, reward=1),
                              oracle=CheckReport("oracle", True, table=oracle, reward=1))


def test_solution_change_justified_only_by_oracle_problems():
    base_only = _report({}, {}, ["base: pass_to_pass test failed: t::p: AssertionError"])
    assert solution_change_justified(base_only) is False
    oracle = _report({}, {}, ["oracle: fail_to_pass test failed: t::f: AssertionError"])
    assert solution_change_justified(oracle) is True
    crashed = _report({}, {}, ["oracle: solve.sh exited with 1 in step R2"])
    assert solution_change_justified(crashed) is True
    env = _report({}, {}, ["oracle: reward=0, expected 1",
                           "oracle: fail_to_pass test failed: t::f: Failed: async def functions are not natively supported."])
    assert solution_change_justified(env) is False


MANIFEST = {"fail_to_pass": ["tests/t.py::f1", "tests/t.py::f2", "tests/t.py::f2_long"],
            "pass_to_pass": ["tests/t.py::p"], "anti_cheat": ["tests/t.py::a"]}
COVERAGE = [{"requirement_id": "R1", "tests": ["tests/t.py::f1"]},
            {"requirement_id": "R2", "tests": ["tests/t.py::f2", "tests/t.py::f2_long", "tests/t.py::p"]}]


def test_attribute_problems_by_covering_tests_and_crashed_step():
    att = attribute_problems([
        "base: reward=1, expected 0",                                    # summary: ignored
        "base: fail_to_pass test passed on original code: tests/t.py::f2_long",
        "oracle: fail_to_pass test failed: tests/t.py::f1: AssertionError: 1 != 2",
        "oracle: solve.sh exited with 1 in step R2",
    ], MANIFEST, COVERAGE)
    assert set(att.by_requirement) == {"R1", "R2"}
    assert att.by_requirement["R1"] == ["oracle: fail_to_pass test failed: tests/t.py::f1: AssertionError: 1 != 2"]
    assert len(att.by_requirement["R2"]) == 2 and att.unattributed == []
    assert att.converged(["R1", "R2", "R3"]) == ["R3"]


def test_unattributed_problems_block_freezing():
    att = attribute_problems(["base: anti_cheat test failed: tests/t.py::a: AssertionError",
                              "oracle: fail_to_pass test failed: tests/t.py::f1: boom"], MANIFEST, COVERAGE)
    assert att.unattributed == ["base: anti_cheat test failed: tests/t.py::a: AssertionError"]
    assert att.by_requirement == {"R1": ["oracle: fail_to_pass test failed: tests/t.py::f1: boom"]}
    assert att.converged(["R1", "R2"]) == []
    iso = attribute_problems(["isolated base/fail_to_pass: reward=1, expected 0"], MANIFEST, COVERAGE)
    assert iso.unattributed and iso.converged(["R1"]) == []


def test_no_recategorisation_helpers_exist():
    import harness.validation.convergence as conv
    import harness.core.brief as brief
    assert not hasattr(conv, "auto_recategorize")
    assert not hasattr(brief, "reclassify_from_base_run")
