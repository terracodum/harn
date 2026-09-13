from harness.providers.python.test_runner import CheckReport
from harness.validation.convergence import VerificationReport, solution_change_justified


def _report(base: dict[str, str], oracle: dict[str, str], problems: list[str]) -> VerificationReport:
    return VerificationReport(ok=False, build_ok=True, problems=problems,
                              base=CheckReport("base", False, table=base, reward=1),
                              oracle=CheckReport("oracle", True, table=oracle, reward=1))


def test_solution_change_justified_only_by_oracle_problems():
    base_only = _report({}, {}, ["base: pass_to_pass test failed: t::p: AssertionError"])
    assert solution_change_justified(base_only) is False
    oracle = _report({}, {}, ["oracle: fail_to_pass test failed: t::f: AssertionError"])
    assert solution_change_justified(oracle) is True
    crashed = _report({}, {}, ["oracle: solve.sh exited with 1 in step R1"])
    assert solution_change_justified(crashed) is True
    env = _report({}, {}, ["oracle: reward=0, expected 1",
                           "oracle: fail_to_pass test failed: t::f: Failed: async def functions are not natively supported."])
    assert solution_change_justified(env) is False


def test_no_recategorisation_helpers_exist():
    import harness.core.brief as brief
    import harness.validation.convergence as conv
    assert not hasattr(conv, "auto_recategorize")
    assert not hasattr(brief, "reclassify_from_base_run")
