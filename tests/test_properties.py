"""Property-based tests (Hypothesis)."""

import pytest

pytest.importorskip("hypothesis")

import sympy as sp  # noqa: E402
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from proofpilot.tools.math_tools import NumberTheory, miller_rabin, safe_parse  # noqa: E402


@settings(max_examples=200, deadline=None)
@given(st.integers(-10**9, 10**9), st.integers(-10**9, 10**9))
def test_gcd_always_verifies(a, b):
    t = NumberTheory()
    args = t.validate({"operation": "gcd", "a": a, "b": b})
    assert t.verify(args, t.run(args)).ok


@settings(max_examples=300, deadline=None)
@given(st.integers(0, 10**6))
def test_miller_rabin_agrees_with_sympy(n):
    assert miller_rabin(n) == bool(sp.isprime(n))


@settings(max_examples=100, deadline=None)
@given(st.integers(2, 10**12))
def test_factorize_always_verifies(n):
    t = NumberTheory()
    args = t.validate({"operation": "factorize", "n": n})
    assert t.verify(args, t.run(args)).ok


@settings(max_examples=300, deadline=None)
@given(st.text(alphabet=list("xyz0123456789+-*/^(). "), max_size=25))
def test_safe_parse_only_raises_valueerror(s):
    try:
        safe_parse(s)
    except ValueError:
        pass
