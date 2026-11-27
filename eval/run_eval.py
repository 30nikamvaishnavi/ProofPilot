"""Evaluation harness.  python -m eval.run_eval [--fault-rate 0.3] [--seed 0] [--langgraph]

Measures, over ~60 problems, what the verify/repair loop buys you:
  * accuracy against hand-checked expected answers
  * how many wrong answers were marked "verified" (should be 0)
  * retries and latency
With --fault-rate > 0, a seeded fault injector corrupts tool outputs to simulate an unreliable solver.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval.problems import build  # noqa: E402
from proofpilot.engine import Runtime, run_engine  # noqa: E402
from proofpilot.faults import make_fault_hook  # noqa: E402
from proofpilot.tools.math_tools import safe_parse  # noqa: E402

X = sp.Symbol("x")


def matches(got, exp) -> bool:
    if isinstance(exp, bool) or isinstance(got, bool):
        return got is exp
    if isinstance(exp, (int, float)) and isinstance(got, (int, float)):
        return math.isclose(got, exp, rel_tol=1e-4, abs_tol=1e-9)
    if isinstance(exp, str):
        if not isinstance(got, str):
            return False
        try:
            a, b = safe_parse(got), safe_parse(exp)
            if exp in ("x**3/3", "sin(x)"):  # indefinite integrals: equal up to a constant
                return sp.simplify(sp.diff(a - b, X)) == 0
            return sp.simplify(a - b) == 0
        except Exception:
            return got == exp
    if isinstance(exp, list):
        return (
            isinstance(got, list)
            and len(got) == len(exp)
            and all(matches(g, e) for g, e in zip(got, exp, strict=True))
        )
    if isinstance(exp, dict):
        return isinstance(got, dict) and all(k in got and matches(got[k], v) for k, v in exp.items())
    return got == exp


def run_mode(problems, verify: bool, fault_rate: float, seed: int, orchestrator: str):
    rt = Runtime(verify_enabled=verify, use_memory=False, fault_hook=make_fault_hook(fault_rate, seed) if fault_rate else None)
    runner = run_engine
    if orchestrator == "langgraph":
        from proofpilot.graph import GraphAgent

        agent = GraphAgent(rt)
        runner = lambda p, r: agent.run(p)  # noqa: E731
    rows = []
    for item in problems:
        t0 = time.perf_counter()
        report = runner(item["problem"], rt)
        ms = (time.perf_counter() - t0) * 1000
        step = report.steps[0] if report.steps else None
        correct = bool(step and step.value is not None and matches(step.value, item["expected"]))
        rows.append({"id": item["id"], "domain": item["domain"], "correct": correct, "verified": bool(step and step.verified),
                     "attempts": step.attempts if step else 0, "ms": ms})
    return rows


def summarise(rows):
    n = len(rows)
    ms = sorted(r["ms"] for r in rows)
    return {
        "accuracy": sum(r["correct"] for r in rows) / n,
        "verified_rate": sum(r["verified"] for r in rows) / n,
        "wrong_but_verified": sum(r["verified"] and not r["correct"] for r in rows),
        "mean_attempts": statistics.mean(r["attempts"] for r in rows),
        "retried": sum(r["attempts"] > 1 for r in rows),
        "p50_ms": statistics.median(ms),
        "p95_ms": ms[min(n - 1, math.ceil(0.95 * n) - 1)],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fault-rate", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--langgraph", action="store_true", help="also run the LangGraph orchestrator")
    ap.add_argument("--out", default=str(Path(__file__).with_name("RESULTS.md")))
    args = ap.parse_args()

    problems = build()
    orchestrators = ["custom"] + (["langgraph"] if args.langgraph else [])
    configs = [("no faults", 0.0), (f"{int(args.fault_rate * 100)}% injected faults", args.fault_rate)]
    lines = [f"# Evaluation results ({len(problems)} problems, seed {args.seed})", "",
             "| orchestrator | condition | verifier | accuracy | verified | wrong but marked verified | mean attempts | retried | p50 ms | p95 ms |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    raw = {}
    for orch in orchestrators:
        for label, rate in configs:
            for verify in (True, False):
                rows = run_mode(problems, verify, rate, args.seed, orch)
                s = summarise(rows)
                raw[f"{orch}|{label}|{'on' if verify else 'off'}"] = s
                lines.append(f"| {orch} | {label} | {'on' if verify else 'off'} | {s['accuracy']:.1%} | {s['verified_rate']:.1%} | {s['wrong_but_verified']} | "
                             f"{s['mean_attempts']:.2f} | {s['retried']} | {s['p50_ms']:.1f} | {s['p95_ms']:.1f} |")
    lines += ["", "Faults are injected by `proofpilot.faults` (seeded). 'verifier off' accepts the first answer. Expected answers are in `eval/problems.py`."]
    Path(args.out).write_text("\n".join(lines) + "\n")
    Path(args.out).with_suffix(".json").write_text(json.dumps(raw, indent=2))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
