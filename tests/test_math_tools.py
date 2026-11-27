import pytest
import sympy as sp

from proofpilot.tools.math_tools import (
    Differentiate,
    Integrate,
    LinearSystem,
    MatrixOps,
    NumberTheory,
    SolveEquation,
    miller_rabin,
    safe_parse,
)


def run(tool, args):
    a = tool.validate(args)
    out = tool.run(a)
    return a, out, tool.verify(a, out)


def same(expr_a, expr_b):
    return sp.simplify(safe_parse(expr_a) - safe_parse(expr_b)) == 0


@pytest.mark.parametrize("eq,roots", [
    ("x^2-5x+6=0", ["2", "3"]),
    ("2x+3=7", ["2"]),
    ("x**2=4", ["-2", "2"]),
    ("x^3-x=0", ["-1", "0", "1"]),
    ("x**2+1=0", ["-I", "I"]),
])
def test_solve_equation(eq, roots):
    _, out, v = run(SolveEquation(), {"equation": eq})
    assert out.value == roots
    assert v.ok, v.detail


def test_solve_no_solution_raises():
    with pytest.raises(ValueError):
        SolveEquation().run(SolveEquation().validate({"equation": "x+1=x"}))


def test_solve_rejects_extra_symbols():
    with pytest.raises(ValueError, match="other unknown"):
        SolveEquation().run(SolveEquation().validate({"equation": "x+y=1"}))


def test_solve_verifier_rejects_wrong_roots():
    t = SolveEquation()
    a, out, _ = run(t, {"equation": "x^2-5x+6=0"})
    out.value = ["2", "5"]
    assert not t.verify(a, out).ok


def test_solve_verifier_rejects_missing_root():
    t = SolveEquation()
    a, out, _ = run(t, {"equation": "x^2-5x+6=0"})
    out.value = ["2"]
    assert not t.verify(a, out).ok


@pytest.mark.parametrize("expr,expected", [
    ("x^3", "3*x**2"), ("sin(x)", "cos(x)"), ("exp(2x)", "2*exp(2*x)"), ("x^2*sin(x)", "2*x*sin(x)+x**2*cos(x)"), ("log(x)", "1/x"),
])
def test_differentiate(expr, expected):
    _, out, v = run(Differentiate(), {"expression": expr})
    assert same(out.value, expected)
    assert v.ok, v.detail


def test_differentiate_other_variable():
    _, out, v = run(Differentiate(), {"expression": "x*y**2", "variable": "y"})
    assert same(out.value, "2*x*y") and v.ok


def test_differentiate_verifier_rejects_wrong():
    t = Differentiate()
    a, out, _ = run(t, {"expression": "sin(x)"})
    out.value = "x"
    assert not t.verify(a, out).ok


def test_integrate_indefinite():
    _, out, v = run(Integrate(), {"expression": "x*exp(x)"})
    assert same(out.value, "(x-1)*exp(x)") and v.ok


@pytest.mark.parametrize("expr,lo,hi,expected", [
    ("x^2", "0", "1", "1/3"), ("sin(x)", "0", "pi", "2"), ("exp(-x)", "0", "oo", "1"), ("1/x", "1", "E", "1"),
])
def test_integrate_definite(expr, lo, hi, expected):
    _, out, v = run(Integrate(), {"expression": expr, "lower": lo, "upper": hi})
    assert same(out.value, expected)
    assert v.ok, v.detail


def test_integrate_needs_both_bounds():
    with pytest.raises(ValueError, match="both"):
        Integrate().run(Integrate().validate({"expression": "x", "lower": "0"}))


def test_integrate_verifier_rejects_wrong():
    t = Integrate()
    a, out, _ = run(t, {"expression": "x^2", "lower": "0", "upper": "1"})
    out.value = "1/2"
    assert not t.verify(a, out).ok


def test_matrix_determinant_and_rank():
    _, out, v = run(MatrixOps(), {"matrix": [[1, 2], [3, 4]], "operation": "determinant"})
    assert out.value == "-2" and v.ok
    _, out, v = run(MatrixOps(), {"matrix": [[1, 2], [2, 4]], "operation": "rank"})
    assert out.value == 1 and v.ok


def test_matrix_eigenvalues_real_and_complex():
    _, out, v = run(MatrixOps(), {"matrix": [[2, 0], [0, 3]], "operation": "eigenvalues"})
    assert out.value == ["2", "3"] and v.ok
    _, out, v = run(MatrixOps(), {"matrix": [[0, -1], [1, 0]], "operation": "eigenvalues"})
    assert sorted(out.value) == ["-I", "I"] and v.ok


def test_matrix_inverse():
    _, out, v = run(MatrixOps(), {"matrix": [[1, 2], [3, 4]], "operation": "inverse"})
    assert out.value == [["-2", "1"], ["3/2", "-1/2"]] and v.ok


def test_matrix_errors():
    t = MatrixOps()
    with pytest.raises(ValueError, match="singular"):
        t.run(t.validate({"matrix": [[1, 2], [2, 4]], "operation": "inverse"}))
    with pytest.raises(ValueError, match="square"):
        t.run(t.validate({"matrix": [[1, 2, 3]], "operation": "determinant"}))
    with pytest.raises(ValueError, match="larger"):
        t.run(t.validate({"matrix": [[1] * 9] * 9, "operation": "rank"}))
    with pytest.raises(ValueError, match="equal length"):
        t.run(t.validate({"matrix": [[1, 2], [3]], "operation": "rank"}))


def test_matrix_rejects_bad_operation():
    with pytest.raises(ValueError):
        MatrixOps().validate({"matrix": [[1]], "operation": "explode"})


def test_matrix_verifier_rejects_wrong_det():
    t = MatrixOps()
    a, out, _ = run(t, {"matrix": [[1, 2], [3, 4]], "operation": "determinant"})
    out.value = "5"
    assert not t.verify(a, out).ok


def test_linear_system():
    _, out, v = run(LinearSystem(), {"A": [[2, 1], [1, 3]], "b": [3, 5]})
    assert out.value == ["4/5", "7/5"] and v.ok


def test_linear_system_errors():
    t = LinearSystem()
    with pytest.raises(ValueError, match="singular"):
        t.run(t.validate({"A": [[1, 2], [2, 4]], "b": [1, 2]}))
    with pytest.raises(ValueError, match="one entry"):
        t.run(t.validate({"A": [[1, 0], [0, 1]], "b": [1]}))


@pytest.mark.parametrize("n,expected", [(2, True), (1, False), (0, False), (97, True), (91, False), (7919, True), (10**9 + 7, True)])
def test_is_prime(n, expected):
    _, out, v = run(NumberTheory(), {"operation": "is_prime", "n": n})
    assert out.value is expected and v.ok


def test_factorize_and_gcd():
    _, out, v = run(NumberTheory(), {"operation": "factorize", "n": 360})
    assert out.value == {"2": 3, "3": 2, "5": 1} and v.ok
    _, out, v = run(NumberTheory(), {"operation": "gcd", "a": 84, "b": 36})
    assert out.value == 12 and v.ok
    _, out, v = run(NumberTheory(), {"operation": "gcd", "a": 0, "b": 0})
    assert out.value == 0 and v.ok


def test_number_theory_errors():
    t = NumberTheory()
    with pytest.raises(ValueError):
        t.run(t.validate({"operation": "factorize", "n": 1}))
    with pytest.raises(ValueError, match="needs"):
        t.run(t.validate({"operation": "gcd", "a": 4}))
    with pytest.raises(ValueError, match="too large"):
        t.run(t.validate({"operation": "is_prime", "n": 10**19}))


def test_miller_rabin_known_pseudoprimes():
    for composite in (561, 1105, 1729, 2047, 3215031751):  # Carmichael / strong pseudoprimes
        assert not miller_rabin(composite)
    for prime in (2, 3, 5, 104729, 2**61 - 1):
        assert miller_rabin(prime)
