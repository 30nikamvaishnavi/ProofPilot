import pytest
from conftest_helpers import FakeLLM

from proofpilot.spec import (
    SPLIT_MARKER,
    SpecError,
    SpecWorkflow,
    TheoremSpec,
    VarSpec,
    check_spec,
    extract_spec,
    generate_module,
    parse_safe,
    validate_spec,
)


@pytest.mark.parametrize("text", [
    "The sum of two even numbers is even.",
    "The sum of two odd numbers is even",
    "The product of two odd integers is odd",
    "The product of an even number and an odd number is even",
    "The square of an odd integer is odd",
    "For all integers a, b: a*b = b*a",
    "For all natural numbers n: n*(n+1) % 2 == 0",
    "For all positive reals x: log(x) <= x - 1",
    "For all reals x, y: (x+y)**2 = x**2 + 2*x*y + y**2",
    "For all integers a, b with a > 0 and b > 0: gcd(a, b) <= a",
])
def test_true_statements_pass_the_randomised_check(text):
    spec = extract_spec(text)
    res = check_spec(spec)
    assert res["passed"] and res["counterexample"] is None and not res["vacuous"], res


@pytest.mark.parametrize("text", [
    "The sum of two odd numbers is odd",
    "The product of two even numbers is odd",
    "For all integers n with n >= 0: n**2 > n",
    "For all reals x: x**2 > x",
    "For all integers a, b: a - b = b - a",
])
def test_false_statements_get_a_counterexample(text):
    res = check_spec(extract_spec(text))
    assert not res["passed"] and res["counterexample"] is not None


def test_check_is_reproducible_with_seed():
    spec = extract_spec("For all integers a, b: a*b = b*a")
    assert check_spec(spec, seed=5) == check_spec(spec, seed=5)


def test_vacuous_assumptions_are_flagged():
    spec = TheoremSpec(name="v", statement="s", variables=[VarSpec(name="n", domain="natural")], assumptions=["n > 1000000"], claim="n > 0")
    res = check_spec(spec)
    assert res["vacuous"] and res["tested"] == 0 and not res["passed"]


@pytest.mark.parametrize("text", [
    "", "   ", "make me a sandwich", "For all widgets w: w = w", "x" * 1600,
    "For all integers n: __import__('os').system('id') == 0",
    "For all integers n: n.__class__ == 1",
    "For all integers n: open('f') == 1",
    "For all integers n: [n] == [n]",
    "For all integers n: 'a' == 'a'",
    "For all integers n: m > 0",
])
def test_unsafe_or_unparseable_text_is_rejected(text):
    with pytest.raises(SpecError):
        extract_spec(text)


def test_validate_spec_limits():
    many = [VarSpec(name=c) for c in "abcdef"]
    with pytest.raises(SpecError, match="at most"):
        validate_spec(TheoremSpec(name="n", statement="s", variables=many, claim="a == a"))
    with pytest.raises(SpecError, match="single lowercase"):
        validate_spec(TheoremSpec(name="n", statement="s", variables=[VarSpec(name="AB")], claim="1 == 1"))
    with pytest.raises(SpecError, match="distinct"):
        validate_spec(TheoremSpec(name="n", statement="s", variables=[VarSpec(name="a"), VarSpec(name="a")], claim="a == a"))
    with pytest.raises(SpecError, match="at least one"):
        validate_spec(TheoremSpec(name="n", statement="s", variables=[], claim="1 == 1"))


def test_parse_safe_blocks_big_powers_at_evaluation_time():
    spec = TheoremSpec(name="p", statement="s", variables=[VarSpec(name="n")], claim="n ** 1000 > 0")
    res = check_spec(spec, n=20)
    assert res["tested"] == 0 and res["skipped_undefined"] == 20
    parse_safe("sqrt(abs(n)) >= 0", {"n"})


def test_undefined_samples_are_skipped_not_crashed():
    spec = TheoremSpec(name="d", statement="s", variables=[VarSpec(name="x", domain="real")], claim="1 / x * x == 1")
    res = check_spec(spec, n=300)
    assert res["passed"]


def test_approval_gate_blocks_code_generation():
    wf = SpecWorkflow()
    s = wf.draft("For all integers a, b: a*b = b*a")
    assert s.status == "draft"
    with pytest.raises(SpecError, match="approved"):
        wf.generate(s.id)
    wf.approve(s.id)
    assert "def claim(a, b)" in wf.generate(s.id)


def test_human_edit_is_revalidated_on_approval():
    wf = SpecWorkflow()
    s = wf.draft("For all integers a, b: a*b = b*a")
    ok = wf.approve(s.id, {"claim": "a + b == b + a"})
    assert ok.claim == "a + b == b + a" and ok.status == "approved"
    s2 = wf.draft("For all integers a, b: a*b = b*a")
    with pytest.raises(SpecError):
        wf.approve(s2.id, {"claim": "__import__('os')"})
    assert wf.get(s2.id).status == "draft"


def test_unknown_spec_id():
    with pytest.raises(KeyError):
        SpecWorkflow().get("nope")


def test_generated_module_compiles_and_claim_function_works():
    wf = SpecWorkflow()
    s = wf.draft("The sum of two even numbers is even")
    wf.approve(s.id)
    src = wf.generate(s.id)
    compile(src, "generated", "exec")
    assert "@given(a=st.integers(-1000, 1000), b=st.integers(-1000, 1000))" in src
    ns: dict = {}
    exec(src.split(SPLIT_MARKER)[0], ns)  # pure-python part only (no hypothesis needed)
    assert ns["assumptions_hold"](2, 4) and not ns["assumptions_hold"](2, 3)
    assert ns["claim"](2, 4) and not ns["claim"](2, 3)


def test_generated_real_domain_uses_isclose_and_imports():
    spec = extract_spec("For all positive reals x: sqrt(x) * sqrt(x) = x")
    spec.status = "approved"
    src = generate_module(spec)
    assert "isclose" in src and "from math import isclose, sqrt" in src
    ns: dict = {}
    exec(src.split(SPLIT_MARKER)[0], ns)
    assert ns["claim"](2.0)


def test_llm_extraction_then_rule_fallback():
    good = {"name": "comm", "variables": [{"name": "a", "domain": "integer"}], "assumptions": [], "claim": "a + 0 == a"}
    s = extract_spec("anything free-form", FakeLLM(good))
    assert s.source == "llm:fake" and s.claim == "a + 0 == a"
    evil = {"name": "x", "variables": [{"name": "a"}], "claim": "__import__('os').system('id')"}
    s = extract_spec("For all integers a: a = a", FakeLLM(evil))  # unsafe LLM output -> rules take over
    assert s.source == "rules"
    s = extract_spec("For all integers a: a = a", FakeLLM(RuntimeError("down")))
    assert s.source == "rules"


def test_check_n_is_clamped_by_workflow():
    wf = SpecWorkflow()
    s = wf.draft("For all integers a: a = a")
    assert wf.check(s.id, n=10**9)["tested"] <= 20000
