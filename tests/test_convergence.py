from harness.providers.python.test_runner import CheckReport
from harness.validation.convergence import VerificationReport, auto_recategorize


def _report(base: dict[str, str], oracle: dict[str, str], problems: list[str]) -> VerificationReport:
    return VerificationReport(ok=False, build_ok=True, problems=problems,
                              base=CheckReport("base", False, table=base, reward=1),
                              oracle=CheckReport("oracle", True, table=oracle, reward=1))


def test_solution_change_justified_only_by_oracle_problems():
    from harness.validation.convergence import solution_change_justified
    base_only = _report({}, {}, ["base: pass_to_pass test failed: t::p: AssertionError"])
    assert solution_change_justified(base_only) is False
    oracle = _report({}, {}, ["oracle: fail_to_pass test failed: t::f: AssertionError"])
    assert solution_change_justified(oracle) is True
    crashed = _report({}, {}, ["oracle: solve.sh exited with 1"])
    assert solution_change_justified(crashed) is True
    env = _report({}, {}, ["oracle: reward=0, expected 1",
                           "oracle: fail_to_pass test failed: t::f: Failed: async def functions are not natively supported."])
    assert solution_change_justified(env) is False


def test_auto_recategorize_moves_leaky_tests():
    manifest = {"fail_to_pass": ["t::real", "t::leaky"], "pass_to_pass": ["t::p"], "anti_cheat": []}
    report = _report({"t::real": "failed", "t::leaky": "passed", "t::p": "passed"},
                     {"t::real": "passed", "t::leaky": "passed", "t::p": "passed"},
                     ["base: fail_to_pass test passed on original code: t::leaky"])
    assert auto_recategorize(report, manifest) == ["t::leaky"]
    assert manifest["fail_to_pass"] == ["t::real"] and manifest["pass_to_pass"] == ["t::p", "t::leaky"]


def test_auto_recategorize_refuses_when_other_problems_or_nothing_left():
    manifest = {"fail_to_pass": ["t::leaky"], "pass_to_pass": [], "anti_cheat": []}
    report = _report({"t::leaky": "passed"}, {"t::leaky": "passed"},
                     ["base: fail_to_pass test passed on original code: t::leaky"])
    assert auto_recategorize(report, manifest) == []          # would empty fail_to_pass
    manifest = {"fail_to_pass": ["t::real", "t::leaky"], "pass_to_pass": [], "anti_cheat": ["t::a"]}
    report = _report({"t::real": "failed", "t::leaky": "passed", "t::a": "failed"},
                     {"t::real": "passed", "t::leaky": "passed", "t::a": "failed"},
                     ["base: fail_to_pass test passed on original code: t::leaky", "base: anti_cheat test failed: t::a"])
    assert auto_recategorize(report, manifest) == []          # anti_cheat problem needs the LLM
    assert manifest["fail_to_pass"] == ["t::real", "t::leaky"]
