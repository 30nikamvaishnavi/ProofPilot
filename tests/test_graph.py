"""LangGraph orchestrator must behave exactly like the framework-free loop. Needs langgraph."""

import os
import tempfile

import pytest

pytest.importorskip("langgraph")

from proofpilot.engine import Runtime, run_engine  # noqa: E402
from proofpilot.faults import make_fault_hook  # noqa: E402
from proofpilot.graph import GraphAgent  # noqa: E402

PROBLEMS = [
    "solve x^2-5x+6=0",
    "integrate sin(x) from 0 to pi",
    "is 97 prime",
    "solve x^2-1=0; then gcd of 84 and 36",
    "solve x + y = 1",
    "tell me a joke",
]


def test_graph_matches_custom_engine():
    for p in PROBLEMS:
        a = GraphAgent(Runtime()).run(p)
        b = run_engine(p, Runtime())
        assert (a.status, a.final_answer) == (b.status, b.final_answer), p
        assert [e.node for e in a.trace] == [e.node for e in b.trace], p


def test_graph_matches_custom_engine_under_faults():
    for p in PROBLEMS[:4]:
        a = GraphAgent(Runtime(fault_hook=make_fault_hook(0.5, seed=2))).run(p)
        b = run_engine(p, Runtime(fault_hook=make_fault_hook(0.5, seed=2)))
        assert [e.node for e in a.trace] == [e.node for e in b.trace], p
        assert a.final_answer == b.final_answer


def test_checkpoint_is_persisted_in_sqlite():
    path = os.path.join(tempfile.mkdtemp(), "ckpt.db")
    agent = GraphAgent(Runtime(), db_path=path)
    agent.run("is 97 prime", thread_id="t1")
    snap = agent.graph.get_state({"configurable": {"thread_id": "t1"}})
    assert snap.values["report"]["status"] == "verified"
    assert os.path.getsize(path) > 0
