import random
import time

import pytest

from proofpilot.tools.math_tools import parse_equation, safe_parse


@pytest.mark.parametrize("bad", [
    "__import__('os').system('ls')", "open(1)", "x.real", "x.__class__", "lambda: 1", "exec(1)", "import os",
    "x; y", "x[0]", "x{1}", "a" * 400, "", "   ", "9**9**9**9", "9^9^9", "x**1000", "12345678901234567890", "x**(2**3)",
    "os.system", "x = = 1 = 2",
])
def test_safe_parse_rejects_dangerous_input(bad):
    with pytest.raises(ValueError):
        if "=" in bad:
            parse_equation(bad)
        else:
            safe_parse(bad)


@pytest.mark.parametrize("good,text", [
    ("5x+1", "5*x + 1"), ("e^x", "exp(x)"), ("2.5*x", "2.5*x"), ("sqrt(x)", "sqrt(x)"), ("x**2*y**3", "x**2*y**3"), ("ln(x)", "log(x)"),
])
def test_safe_parse_accepts_normal_math(good, text):
    assert str(safe_parse(good)) == text


def test_parse_equation_moves_everything_left():
    assert str(parse_equation("x+1=3")) == "x - 2"
    assert str(parse_equation("x**2")) == "x**2"


def test_safe_parse_fuzz_only_raises_valueerror_and_is_fast():
    rng = random.Random(7)
    alphabet = list("xyz0123456789+-*/^(). ") + ["sin", "cos", "exp", "log", "sqrt", "pi", "oo", "**", "=", "__", "'", "e"]
    start = time.time()
    for _ in range(1500):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 18)))
        try:
            safe_parse(s)
        except ValueError:
            pass
    assert time.time() - start < 30
