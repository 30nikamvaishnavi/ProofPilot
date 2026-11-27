"""Command line: ``proofpilot "solve x^2-5x+6=0"``"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .engine import Runtime, run_engine
from .llm import get_llm
from .memory import MemoryStore
from .spec import SpecError, SpecWorkflow


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="proofpilot", description="Verified multi-step STEM agent")
    sub = p.add_subparsers(dest="cmd")

    solve = sub.add_parser("solve", help="solve a problem (default command)")
    solve.add_argument("problem")
    solve.add_argument("--json", action="store_true", help="print the full report as JSON")
    solve.add_argument("--trace", action="store_true", help="print the step-by-step trace")
    solve.add_argument("--langgraph", action="store_true", help="use the LangGraph orchestrator")

    spec = sub.add_parser("spec", help="theorem text -> spec -> randomised check -> test module")
    spec.add_argument("text")
    spec.add_argument("--approve", action="store_true", help="approve the spec and print generated code")
    spec.add_argument("--n", type=int, default=1000)

    serve = sub.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0] not in {"solve", "spec", "serve", "-h", "--help"}:
        args_list.insert(0, "solve")
    args = p.parse_args(args_list)
    if args.cmd is None:
        p.print_help()
        return 2

    llm = get_llm()
    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("proofpilot.api:app", host=args.host, port=args.port)
        return 0

    if args.cmd == "spec":
        wf = SpecWorkflow(llm)
        try:
            s = wf.draft(args.text)
            print(json.dumps(s.model_dump(), indent=2))
            print("check:", json.dumps(wf.check(s.id, args.n)))
            if args.approve:
                wf.approve(s.id)
                print(wf.generate(s.id))
        except SpecError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0

    rt = Runtime(memory=MemoryStore(os.environ.get("PROOFPILOT_DB", ":memory:")), llm=llm)
    if args.langgraph:
        from .graph import GraphAgent

        report = GraphAgent(rt).run(args.problem)
    else:
        report = run_engine(args.problem, rt)
    if args.json:
        print(json.dumps(report.model_dump(), indent=2, default=str))
    else:
        print(f"[{report.status.upper()}] confidence {report.confidence:.2f}  planner={report.planner}")
        print(report.final_answer)
        if args.trace:
            print("\ntrace:")
            for e in report.trace:
                print(f"  {e.t:7.3f}s {e.node:<8} {e.message}")
    return 0 if report.status == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
