"""Mutation test: for every tool, corrupt a correct answer and require the verifier to reject it."""

import pytest

from proofpilot.faults import corrupt_value
from proofpilot.tools.base import ToolOutput, default_registry

GOLDEN = [
    ("solve_equation", {"equation": "x^2-5x+6=0"}),
    ("differentiate", {"expression": "x^2*sin(x)"}),
    ("integrate", {"expression": "x*exp(x)"}),
    ("integrate", {"expression": "sin(x)", "lower": "0", "upper": "pi"}),
    ("matrix_ops", {"matrix": [[1, 2], [3, 4]], "operation": "determinant"}),
    ("matrix_ops", {"matrix": [[1, 2], [2, 4]], "operation": "rank"}),
    ("matrix_ops", {"matrix": [[2, 0], [0, 3]], "operation": "eigenvalues"}),
    ("matrix_ops", {"matrix": [[1, 2], [3, 4]], "operation": "inverse"}),
    ("linear_system", {"A": [[2, 1], [1, 3]], "b": [3, 5]}),
    ("number_theory", {"operation": "is_prime", "n": 97}),
    ("number_theory", {"operation": "factorize", "n": 360}),
    ("number_theory", {"operation": "gcd", "a": 84, "b": 36}),
    ("shortest_path", {"edges": [["A", "B", 4], ["A", "C", 1], ["C", "B", 2], ["B", "D", 5]], "source": "A", "target": "D"}),
    ("minimum_spanning_tree", {"edges": [["A", "B", 4], ["A", "C", 1], ["C", "B", 2], ["B", "D", 5]]}),
    ("topological_sort", {"edges": [["a", "b"], ["b", "c"], ["a", "c"]]}),
    ("black_scholes", {"S": 100, "K": 105, "T": 1, "r": 0.05, "sigma": 0.2, "option_type": "call"}),
    ("portfolio_variance", {"weights": [0.5, 0.5], "cov": [[0.04, 0.01], [0.01, 0.09]]}),
]


@pytest.mark.parametrize("name,args", GOLDEN)
def test_verifier_accepts_correct_answer(name, args):
    tool = default_registry().get(name)
    parsed = tool.validate(args)
    assert tool.verify(parsed, tool.run(parsed)).ok


@pytest.mark.parametrize("name,args", GOLDEN)
def test_verifier_rejects_corrupted_answer(name, args):
    tool = default_registry().get(name)
    parsed = tool.validate(args)
    good = tool.run(parsed)
    bad = ToolOutput(corrupt_value(good.value), good.display)
    assert bad.value != good.value
    try:
        accepted = tool.verify(parsed, bad).ok  # a crash inside the verifier also counts as rejection (engine does the same)
    except Exception:
        accepted = False
    assert not accepted
