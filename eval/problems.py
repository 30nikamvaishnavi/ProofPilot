"""60 evaluation problems. Expected answers are hand-written or computed with an independent
implementation (math.erf for Black-Scholes, plain Python for variance), never by ProofPilot itself."""

from __future__ import annotations

import math

G1 = [["A", "B", 4], ["A", "C", 1], ["C", "B", 2], ["B", "D", 5]]
G2 = [["S", "A", 7], ["S", "B", 2], ["B", "A", 3], ["A", "T", 1], ["B", "T", 6]]
G3 = [[1, 2, 1], [2, 3, 2], [1, 3, 3], [3, 4, 4]]


def _bs(s, k, t, r, sig, kind):
    n = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))  # noqa: E731
    d1 = (math.log(s / k) + (r + sig * sig / 2) * t) / (sig * math.sqrt(t))
    d2 = d1 - sig * math.sqrt(t)
    if kind == "call":
        return s * n(d1) - k * math.exp(-r * t) * n(d2)
    return k * math.exp(-r * t) * n(-d2) - s * n(-d1)


def _pv(w, c):
    return sum(w[i] * w[j] * c[i][j] for i in range(len(w)) for j in range(len(w)))


def build() -> list[dict]:
    P: list[tuple[str, str, object]] = []  # (domain, problem, expected value)
    m = lambda text, exp: P.append(("math", text, exp))  # noqa: E731
    # solve
    m("solve x^2-5x+6=0", ["2", "3"])
    m("solve 2x+3=7", ["2"])
    m("solve x^2=9", ["-3", "3"])
    m("solve x^2-1=0", ["-1", "1"])
    m("solve x^2+4=0", ["-2*I", "2*I"])
    m("solve 3x-9=0", ["3"])
    m("solve x^3-x=0", ["-1", "0", "1"])
    m("solve x^2-2x+1=0", ["1"])
    m("solve (x-1)*(x-2)*(x-3)=0", ["1", "2", "3"])
    # differentiate
    m("differentiate x^3", "3*x**2")
    m("differentiate sin(x)", "cos(x)")
    m("differentiate exp(2x)", "2*exp(2*x)")
    m("differentiate x^2*sin(x)", "2*x*sin(x) + x**2*cos(x)")
    m("differentiate log(x)", "1/x")
    m("differentiate x^4+x", "4*x**3 + 1")
    m("differentiate cos(x)^2", "-2*sin(x)*cos(x)")
    m("differentiate sqrt(x)", "1/(2*sqrt(x))")
    # integrate
    m("integrate x^2 from 0 to 1", "1/3")
    m("integrate sin(x) from 0 to pi", "2")
    m("integrate exp(x) from 0 to 1", "E - 1")
    m("integrate 1/x from 1 to e", "1")
    m("integrate x^3 from 0 to 2", "4")
    m("integrate cos(x) from 0 to pi/2", "1")
    m("integrate exp(-x) from 0 to oo", "1")
    m("integrate x^2", "x**3/3")
    m("integrate cos(x)", "sin(x)")
    # matrices / linear systems
    m("determinant of [[1,2],[3,4]]", "-2")
    m("determinant of [[2,0],[0,3]]", "6")
    m("determinant of [[1,2,3],[4,5,6],[7,8,10]]", "-3")
    m("rank of [[1,2],[2,4]]", 1)
    m("rank of [[1,0],[0,1]]", 2)
    m("eigenvalues of [[2,0],[0,3]]", ["2", "3"])
    m("eigenvalues of [[4,1],[2,3]]", ["2", "5"])
    m("inverse of [[1,2],[3,4]]", [["-2", "1"], ["3/2", "-1/2"]])
    m("inverse of [[2,0],[0,4]]", [["1/2", "0"], ["0", "1/4"]])
    m("solve A=[[2,1],[1,3]] b=[3,5]", ["4/5", "7/5"])
    m("solve A=[[1,1],[1,-1]] b=[3,1]", ["2", "1"])
    # number theory
    for n, prime in ((2, True), (97, True), (7919, True), (91, False), (1000003, True), (561, False)):
        m(f"is {n} prime", prime)
    m("factorize 360", {"2": 3, "3": 2, "5": 1})
    m("factorize 97", {"97": 1})
    m("factorize 1001", {"7": 1, "11": 1, "13": 1})
    m("gcd of 84 and 36", 12)
    m("gcd of 17 and 5", 1)
    m("gcd of 100 and 75", 25)
    # computer science
    P.append(("cs", f"shortest path from A to D in {G1}", {"cost": 8.0, "path": ["A", "C", "B", "D"]}))
    P.append(("cs", f"shortest path from S to T in {G2}", {"cost": 6.0, "path": ["S", "B", "A", "T"]}))
    P.append(("cs", f"minimum spanning tree of {G1}", {"total_weight": 8.0}))
    P.append(("cs", f"minimum spanning tree of {G3}", {"total_weight": 7.0}))
    P.append(("cs", "topological order of [['a','b'],['b','c'],['a','c']]", ["a", "b", "c"]))
    P.append(("cs", "topological order of [['pants','shoes'],['pants','belt'],['shirt','tie'],['tie','jacket']]", ["pants", "belt", "shirt", "shoes", "tie", "jacket"]))
    # finance
    for (s, k, t, r, sig, kind) in [(100, 100, 1, 0.05, 0.2, "call"), (100, 100, 1, 0.05, 0.2, "put"), (100, 105, 1, 0.05, 0.2, "call"),
                                    (50, 45, 0.5, 0.03, 0.25, "call"), (80, 90, 2, 0.01, 0.35, "put")]:
        P.append(("finance", f"price a {kind} S={s} K={k} T={t} r={r} sigma={sig}", {"price": round(_bs(s, k, t, r, sig, kind), 6)}))
    for w, c in [([0.5, 0.5], [[0.04, 0.01], [0.01, 0.09]]), ([1, 0], [[0.04, 0.01], [0.01, 0.09]]), ([0.6, 0.4], [[0.04, 0.01], [0.01, 0.09]]),
                 ([0.2, 0.3, 0.5], [[0.04, 0.006, 0], [0.006, 0.09, 0.01], [0, 0.01, 0.16]])]:
        P.append(("finance", f"portfolio variance weights={w} cov={c}", {"variance": _pv(w, c)}))
    return [{"id": i + 1, "domain": d, "problem": p, "expected": e} for i, (d, p, e) in enumerate(P)]
