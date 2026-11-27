"""Math tools (SymPy primary solver, NumPy/SciPy/independent code as verifier)."""

from __future__ import annotations

import math
import re
from typing import Any, Literal

import numpy as np
import sympy as sp
from pydantic import BaseModel, Field
from scipy import integrate as sp_integrate
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from ..schemas import VerifyResult
from .base import Tool, ToolOutput

# --------------------------------------------------------------------------------------
# Safe expression parsing (never sympify raw user text: it is eval-based)
# --------------------------------------------------------------------------------------
_FUNCS: dict[str, Any] = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "cot": sp.cot, "sec": sp.sec, "csc": sp.csc,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "exp": sp.exp, "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt, "abs": sp.Abs, "Abs": sp.Abs,
}
_CONSTS: dict[str, Any] = {"pi": sp.pi, "E": sp.E, "I": sp.I, "oo": sp.oo}
_LOCALS = {**_FUNCS, **_CONSTS}
_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
_ALLOWED_CHARS = re.compile(r"^[A-Za-z0-9\s\+\-\*/\^\(\)\.,]*$")
MAX_EXPR_LEN = 300


def _guard_cost(s: str) -> None:
    """Reject inputs that would make eager evaluation explode (DoS protection)."""
    flat = s.replace("^", "**")
    if re.search(r"\*\*\s*[\w\(\)\.]+\s*\*\*", flat):
        raise ValueError("nested powers (power towers) are not allowed")
    for m in re.finditer(r"\*\*\s*\(?\s*(\d+)", flat):
        if int(m.group(1)) > 100:
            raise ValueError("exponents above 100 are not allowed")
    if re.search(r"\d{19,}", flat):
        raise ValueError("numbers with more than 18 digits are not allowed")


def safe_parse(text: str) -> sp.Expr:
    """Parse a math expression with a character whitelist and identifier whitelist."""
    s = str(text).strip().replace("==", "=")
    if not s:
        raise ValueError("empty expression")
    if len(s) > MAX_EXPR_LEN:
        raise ValueError(f"expression longer than {MAX_EXPR_LEN} characters")
    if not _ALLOWED_CHARS.match(s):
        raise ValueError("expression contains characters that are not allowed")
    if re.search(r"[A-Za-z0-9\)]\s*\.\s*[A-Za-z_]", s):
        raise ValueError("attribute access is not allowed")
    s = re.sub(r"\be\b", "E", s)
    _guard_cost(s)
    for token in re.findall(r"[A-Za-z]+", s):
        if token in _LOCALS:
            continue
        if len(token) > 3:
            raise ValueError(f"unknown name '{token}'")
    try:
        expr = parse_expr(s, local_dict=dict(_LOCALS), transformations=_TRANSFORMS, evaluate=True)
    except Exception as exc:  # sympy raises many exception types
        raise ValueError(f"could not parse '{text}': {exc}") from exc
    if not isinstance(expr, sp.Basic):
        raise ValueError("expression did not parse to a symbolic value")
    return expr


def parse_equation(text: str) -> sp.Expr:
    """Return lhs - rhs for 'lhs = rhs' (or the expression itself if there is no '=')."""
    s = str(text).replace("==", "=")
    if s.count("=") > 1:
        raise ValueError("equation may contain only one '='")
    if "=" in s:
        lhs, rhs = s.split("=", 1)
        return safe_parse(lhs) - safe_parse(rhs)
    return safe_parse(s)


def _symbol(name: str) -> sp.Symbol:
    if not re.fullmatch(r"[A-Za-z]", name or ""):
        raise ValueError("variable must be a single letter")
    return sp.Symbol(name)


_SAMPLES = (0.5, 0.9, 1.3, 1.7, 2.3, 3.1)


def _numeric_fn(expr: sp.Expr, var: sp.Symbol):
    """Lambdify with any other free symbols pinned to fixed numbers."""
    others = sorted(expr.free_symbols - {var}, key=str)
    pins = {s: 1.3 + 0.4 * i for i, s in enumerate(others)}
    return sp.lambdify(var, expr.subs(pins), modules=["numpy"])


def _finite_real(x: Any) -> float | None:
    try:
        c = complex(x)
    except (TypeError, ValueError):
        return None
    if abs(c.imag) > 1e-9 or not math.isfinite(c.real):
        return None
    return c.real


def _close(a: float, b: float, rel: float = 1e-4, abs_: float = 1e-6) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_)


# --------------------------------------------------------------------------------------
class SolveArgs(BaseModel):
    equation: str = Field(max_length=MAX_EXPR_LEN)
    variable: str = "x"


class SolveEquation(Tool):
    name = "solve_equation"
    domain = "math"
    description = "Solve a single-variable algebraic or trigonometric equation exactly."
    arg_hint = 'equation: str like "x**2-5*x+6=0"; variable: str = "x"'
    assumptions = ["Equations are solved over the complex numbers; all roots returned by SymPy are listed."]
    ArgsModel = SolveArgs

    def run(self, args: SolveArgs) -> ToolOutput:
        var = _symbol(args.variable)
        expr = parse_equation(args.equation)
        extra = expr.free_symbols - {var}
        if extra:
            raise ValueError(f"equation has other unknown symbols: {sorted(map(str, extra))}")
        sols = sp.solve(expr, var)
        if not sols:
            raise ValueError("no solutions found")
        sols = sorted(sols, key=lambda s: (float(sp.re(sp.N(s))), float(sp.im(sp.N(s)))))
        values = [str(s) for s in sols]
        return ToolOutput(values, ", ".join(f"{var} = {v}" for v in values))

    def verify(self, args: SolveArgs, output: ToolOutput) -> VerifyResult:
        var = _symbol(args.variable)
        expr = parse_equation(args.equation)
        sols = [safe_parse(v) for v in output.value]
        for s in sols:
            residual = abs(complex(sp.N(expr.subs(var, s), 30)))
            if residual > 1e-8:
                return VerifyResult(ok=False, method="back-substitution", detail=f"residual {residual:.3g} at {s}")
        detail = "all roots satisfy the equation (residual < 1e-8)"
        method = "back-substitution"
        poly = None
        try:
            poly = sp.Poly(sp.expand(expr), var)
        except sp.PolynomialError:
            pass
        if poly is not None and 1 <= poly.degree() <= 10 and all(c.is_number for c in poly.all_coeffs()):
            coeffs = [complex(sp.N(c)) for c in poly.all_coeffs()]
            roots = np.roots(coeffs)
            numeric = [complex(sp.N(s)) for s in sols]
            for r in roots:
                if not any(abs(r - n) < 1e-3 for n in numeric):
                    return VerifyResult(ok=False, method="numpy.roots", detail=f"missing root near {r}")
            method = "back-substitution + numpy.roots"
            detail += "; numpy.roots finds no additional roots"
        return VerifyResult(ok=True, method=method, detail=detail)


# --------------------------------------------------------------------------------------
class DiffArgs(BaseModel):
    expression: str = Field(max_length=MAX_EXPR_LEN)
    variable: str = "x"


class Differentiate(Tool):
    name = "differentiate"
    domain = "math"
    description = "Differentiate an expression symbolically with respect to one variable."
    arg_hint = 'expression: str like "x**2*sin(x)"; variable: str = "x"'
    assumptions = ["Functions are differentiated symbolically; other symbols are treated as constants."]
    ArgsModel = DiffArgs

    def run(self, args: DiffArgs) -> ToolOutput:
        var = _symbol(args.variable)
        expr = safe_parse(args.expression)
        d = sp.simplify(sp.diff(expr, var))
        return ToolOutput(str(d), f"d/d{var}[{args.expression}] = {d}")

    def verify(self, args: DiffArgs, output: ToolOutput) -> VerifyResult:
        var = _symbol(args.variable)
        f = _numeric_fn(safe_parse(args.expression), var)
        g = _numeric_fn(safe_parse(output.value), var)
        h = 1e-5
        checked = 0
        for x in _SAMPLES:
            with np.errstate(all="ignore"):
                num = _finite_real((f(x + h) - f(x - h)) / (2 * h))
                sym = _finite_real(g(x))
            if num is None or sym is None:
                continue
            checked += 1
            if not _close(num, sym, rel=1e-4, abs_=1e-5):
                return VerifyResult(ok=False, method="central finite difference", detail=f"at x={x}: {num} vs {sym}")
        if checked < 2:
            return VerifyResult(ok=False, method="central finite difference", detail="too few valid sample points")
        return VerifyResult(ok=True, method="central finite difference", detail=f"matched at {checked} sample points")


# --------------------------------------------------------------------------------------
class IntegrateArgs(BaseModel):
    expression: str = Field(max_length=MAX_EXPR_LEN)
    variable: str = "x"
    lower: str | None = None
    upper: str | None = None


class Integrate(Tool):
    name = "integrate"
    domain = "math"
    description = "Indefinite or definite integral (give both lower and upper for a definite integral)."
    arg_hint = 'expression: str; variable: str = "x"; lower/upper: optional str bounds like "0", "pi", "oo"'
    assumptions = ["Integrals must have a closed form; otherwise the step fails rather than guessing."]
    ArgsModel = IntegrateArgs

    @staticmethod
    def _definite(args: IntegrateArgs) -> bool:
        if (args.lower is None) != (args.upper is None):
            raise ValueError("provide both lower and upper bounds, or neither")
        return args.lower is not None

    def run(self, args: IntegrateArgs) -> ToolOutput:
        var = _symbol(args.variable)
        expr = safe_parse(args.expression)
        if self._definite(args):
            lo, hi = safe_parse(args.lower or ""), safe_parse(args.upper or "")
            res = sp.integrate(expr, (var, lo, hi))
            if res.has(sp.Integral):
                raise ValueError("no closed form found")
            res = sp.simplify(res)
            approx = float(sp.N(res)) if res.is_real else None
            shown = f"{res}" + (f" ≈ {approx:.8g}" if approx is not None else "")
            return ToolOutput(str(res), f"∫[{args.lower},{args.upper}] {args.expression} d{var} = {shown}")
        res = sp.integrate(expr, var)
        if res.has(sp.Integral):
            raise ValueError("no closed form found")
        res = sp.simplify(res)
        return ToolOutput(str(res), f"∫ {args.expression} d{var} = {res} + C")

    def verify(self, args: IntegrateArgs, output: ToolOutput) -> VerifyResult:
        var = _symbol(args.variable)
        integrand = _numeric_fn(safe_parse(args.expression), var)
        result = safe_parse(output.value)
        if self._definite(args):
            lo = float(sp.N(safe_parse(args.lower or "")))
            hi = float(sp.N(safe_parse(args.upper or "")))
            with np.errstate(all="ignore"):
                val, err = sp_integrate.quad(lambda t: float(np.real(integrand(t))), lo, hi, limit=200)
            sym = _finite_real(sp.N(result))
            if sym is None:
                return VerifyResult(ok=False, method="scipy.integrate.quad", detail="result is not a finite real number")
            ok = _close(val, sym, rel=1e-5, abs_=max(1e-6, 10 * err))
            return VerifyResult(ok=ok, method="scipy.integrate.quad", detail=f"quad={val:.10g}, symbolic={sym:.10g}")
        anti = _numeric_fn(result, var)
        h = 1e-5
        checked = 0
        for x in _SAMPLES:
            with np.errstate(all="ignore"):
                num = _finite_real((anti(x + h) - anti(x - h)) / (2 * h))
                orig = _finite_real(integrand(x))
            if num is None or orig is None:
                continue
            checked += 1
            if not _close(num, orig, rel=1e-4, abs_=1e-5):
                return VerifyResult(ok=False, method="differentiate the antiderivative numerically", detail=f"at x={x}: {num} vs {orig}")
        if checked < 2:
            return VerifyResult(ok=False, method="differentiate the antiderivative numerically", detail="too few valid sample points")
        return VerifyResult(ok=True, method="differentiate the antiderivative numerically", detail=f"matched at {checked} sample points")


# --------------------------------------------------------------------------------------
MAX_DIM = 8


def _to_sympy_matrix(rows: list[list[float]]) -> sp.Matrix:
    if not rows or not all(isinstance(r, list) and r for r in rows):
        raise ValueError("matrix must be a non-empty list of non-empty rows")
    if len({len(r) for r in rows}) != 1:
        raise ValueError("matrix rows must have equal length")
    if len(rows) > MAX_DIM or len(rows[0]) > MAX_DIM:
        raise ValueError(f"matrix larger than {MAX_DIM}x{MAX_DIM} is not supported")
    return sp.Matrix([[sp.Rational(str(x)) for x in r] for r in rows])


class MatrixArgs(BaseModel):
    matrix: list[list[float]]
    operation: Literal["determinant", "rank", "eigenvalues", "inverse"]


class MatrixOps(Tool):
    name = "matrix_ops"
    domain = "math"
    description = "Exact determinant, rank, eigenvalues or inverse of a small matrix."
    arg_hint = 'matrix: list of rows e.g. [[1,2],[3,4]]; operation: "determinant"|"rank"|"eigenvalues"|"inverse"'
    assumptions = ["Matrices are at most 8x8 and are converted to exact rationals before solving."]
    ArgsModel = MatrixArgs

    def run(self, args: MatrixArgs) -> ToolOutput:
        m = _to_sympy_matrix(args.matrix)
        op = args.operation
        if op == "determinant":
            if not m.is_square:
                raise ValueError("determinant needs a square matrix")
            d = m.det()
            return ToolOutput(str(d), f"det = {d}")
        if op == "rank":
            r = int(m.rank())
            return ToolOutput(r, f"rank = {r}")
        if not m.is_square:
            raise ValueError(f"{op} needs a square matrix")
        if op == "inverse":
            if m.det() == 0:
                raise ValueError("matrix is singular, no inverse exists")
            inv = m.inv()
            rows = [[str(v) for v in inv.row(i)] for i in range(inv.rows)]
            return ToolOutput(rows, f"inverse = {rows}")
        eig = m.eigenvals()
        values = [str(k) for k, mult in sorted(eig.items(), key=lambda kv: str(kv[0])) for _ in range(mult)]
        return ToolOutput(values, "eigenvalues = " + ", ".join(values))

    def verify(self, args: MatrixArgs, output: ToolOutput) -> VerifyResult:
        a = np.array(args.matrix, dtype=float)
        op = args.operation
        if op == "determinant":
            ref = float(np.linalg.det(a))
            got = float(sp.N(safe_parse(output.value)))
            return VerifyResult(ok=_close(ref, got, rel=1e-6, abs_=1e-8), method="numpy.linalg.det", detail=f"numpy={ref:.10g}, exact={got:.10g}")
        if op == "rank":
            ref = int(np.linalg.matrix_rank(a))
            return VerifyResult(ok=ref == output.value, method="numpy.linalg.matrix_rank", detail=f"numpy={ref}, exact={output.value}")
        if op == "inverse":
            inv = np.array([[float(sp.N(safe_parse(v))) for v in row] for row in output.value])
            err = float(np.max(np.abs(a @ inv - np.eye(a.shape[0]))))
            return VerifyResult(ok=err < 1e-8, method="A @ A^-1 == I (numpy)", detail=f"max error {err:.3g}")
        got_eigenvalues = sorted(
            (complex(sp.N(safe_parse(v))) for v in output.value),
            key=lambda z: (round(z.real, 6), round(z.imag, 6)),
        )
        ref_eigenvalues = sorted(
            np.linalg.eigvals(a),
            key=lambda z: (round(z.real, 6), round(z.imag, 6)),
        )
        ok = len(got_eigenvalues) == len(ref_eigenvalues) and all(
            abs(got - ref) < 1e-5
            for got, ref in zip(got_eigenvalues, ref_eigenvalues, strict=True)
        )
        return VerifyResult(ok=ok, method="numpy.linalg.eigvals", detail="eigenvalue multisets compared with tolerance 1e-5")


# --------------------------------------------------------------------------------------
class LinearSystemArgs(BaseModel):
    A: list[list[float]]
    b: list[float]


class LinearSystem(Tool):
    name = "linear_system"
    domain = "math"
    description = "Solve A x = b for a square, non-singular system."
    arg_hint = "A: list of rows; b: list of numbers"
    assumptions = ["The system must be square and have a unique solution."]
    ArgsModel = LinearSystemArgs

    def run(self, args: LinearSystemArgs) -> ToolOutput:
        a = _to_sympy_matrix(args.A)
        if not a.is_square:
            raise ValueError("A must be square")
        if len(args.b) != a.rows:
            raise ValueError("b must have one entry per row of A")
        if a.det() == 0:
            raise ValueError("A is singular, no unique solution")
        b = sp.Matrix([sp.Rational(str(v)) for v in args.b])
        x = a.LUsolve(b)
        values = [str(v) for v in x]
        return ToolOutput(values, "x = [" + ", ".join(values) + "]")

    def verify(self, args: LinearSystemArgs, output: ToolOutput) -> VerifyResult:
        a = np.array(args.A, dtype=float)
        b = np.array(args.b, dtype=float)
        x = np.array([float(sp.N(safe_parse(v))) for v in output.value])
        ref = np.linalg.solve(a, b)
        resid = float(np.max(np.abs(a @ x - b)))
        ok = bool(np.allclose(x, ref, rtol=1e-6, atol=1e-8)) and resid < 1e-8
        return VerifyResult(ok=ok, method="numpy.linalg.solve + residual", detail=f"max residual {resid:.3g}")


# --------------------------------------------------------------------------------------
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def miller_rabin(n: int) -> bool:
    """Deterministic for n < 3.3e24 (independent of SymPy)."""
    if n < 2:
        return False
    for p in _MR_BASES:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _MR_BASES:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


class NumberTheoryArgs(BaseModel):
    operation: Literal["is_prime", "factorize", "gcd"]
    n: int | None = None
    a: int | None = None
    b: int | None = None


class NumberTheory(Tool):
    name = "number_theory"
    domain = "math"
    description = "Primality test, prime factorisation, or gcd of two integers."
    arg_hint = 'operation: "is_prime"|"factorize"|"gcd"; n: int (for is_prime/factorize); a, b: int (for gcd)'
    assumptions = ["Integers are limited to 10**18 in absolute value."]
    ArgsModel = NumberTheoryArgs
    LIMIT = 10**18

    def _need(self, args: NumberTheoryArgs) -> tuple[int, ...]:
        if args.operation == "gcd":
            if args.a is None or args.b is None:
                raise ValueError("gcd needs integers a and b")
            vals: tuple[int, ...] = (args.a, args.b)
        else:
            if args.n is None:
                raise ValueError(f"{args.operation} needs integer n")
            vals = (args.n,)
        if any(abs(v) > self.LIMIT for v in vals):
            raise ValueError("integer too large (limit 10**18)")
        return vals

    def run(self, args: NumberTheoryArgs) -> ToolOutput:
        vals = self._need(args)
        if args.operation == "is_prime":
            res = bool(sp.isprime(vals[0]))
            return ToolOutput(res, f"{vals[0]} is {'prime' if res else 'not prime'}")
        if args.operation == "gcd":
            g = int(sp.igcd(vals[0], vals[1]))
            return ToolOutput(g, f"gcd({vals[0]}, {vals[1]}) = {g}")
        n = vals[0]
        if n < 2:
            raise ValueError("factorize needs n >= 2")
        fac = sp.factorint(n)
        flat = {str(p): int(e) for p, e in sorted(fac.items())}
        text = " * ".join(f"{p}^{e}" if e > 1 else p for p, e in flat.items())
        return ToolOutput(flat, f"{n} = {text}")

    def verify(self, args: NumberTheoryArgs, output: ToolOutput) -> VerifyResult:
        vals = self._need(args)
        if args.operation == "is_prime":
            ok = miller_rabin(vals[0]) == output.value
            return VerifyResult(ok=ok, method="deterministic Miller-Rabin", detail="independent primality test agrees" if ok else "disagrees")
        if args.operation == "gcd":
            g, a, b = output.value, vals[0], vals[1]
            if g == 0:
                ok = a == 0 and b == 0
            else:
                ok = a % g == 0 and b % g == 0 and math.gcd(a // g, b // g) == 1
            return VerifyResult(ok=ok, method="divisibility and coprime cofactors", detail=f"g={g}")
        n = vals[0]
        product = 1
        for p, e in output.value.items():
            if not miller_rabin(int(p)):
                return VerifyResult(ok=False, method="product + Miller-Rabin", detail=f"{p} is not prime")
            product *= int(p) ** int(e)
        return VerifyResult(ok=product == n, method="product + Miller-Rabin", detail=f"product of factors = {product}")


__all__ = [
    "safe_parse", "parse_equation", "miller_rabin",
    "SolveEquation", "Differentiate", "Integrate", "MatrixOps", "LinearSystem", "NumberTheory",
]
