"""LangGraph orchestration: the same nodes and routers as ``engine.py`` wired into a StateGraph.

Adds what the plain loop does not have: a checkpointer, so a run can be resumed by ``thread_id``
(``SqliteSaver`` when a DB path is given; no persistent checkpointer for ephemeral runs).
"""

from __future__ import annotations

import sqlite3
import uuid
from typing import Any

from langgraph.graph import END as LG_END
from langgraph.graph import START, StateGraph

from .engine import NODES, ROUTERS, AgentState, Runtime, initial_state, json_safe
from .schemas import Plan, Report

# routers return one of these node names
_TARGETS = {
    "plan": ["solve", "finalize"],
    "solve": ["commit", "verify"],
    "verify": ["commit", "repair"],
    "repair": ["commit", "solve"],
    "commit": ["solve", "finalize"],
}


def make_checkpointer(db_path: str | None = None) -> Any:
    if not db_path:
        return None
    from langgraph.checkpoint.sqlite import SqliteSaver

    return SqliteSaver(sqlite3.connect(db_path, check_same_thread=False))


def build_graph(runtime: Runtime, checkpointer: Any | None = None) -> Any:
    graph = StateGraph(AgentState)

    def bind_node(fn: Any) -> Any:
        def node(state: AgentState) -> dict[str, Any]:
            return json_safe(fn(state, runtime))

        return node

    def bind_router(fn: Any) -> Any:
        def router(state: AgentState) -> str:
            return fn(state, runtime)

        return router

    for name, fn in NODES.items():
        graph.add_node(name, bind_node(fn))
    graph.add_edge(START, "plan")
    for name, targets in _TARGETS.items():
        graph.add_conditional_edges(name, bind_router(ROUTERS[name]), {t: t for t in targets})
    graph.add_edge("finalize", LG_END)
    return graph.compile(checkpointer=checkpointer)


class GraphAgent:
    """Convenience wrapper: ``GraphAgent(runtime).run("solve x^2-1=0")``."""

    def __init__(self, runtime: Runtime | None = None, db_path: str | None = None) -> None:
        self.runtime = runtime or Runtime()
        self.graph = build_graph(self.runtime, make_checkpointer(db_path))

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": 200}

    def run(self, problem: str, plan: Plan | None = None, thread_id: str | None = None) -> Report:
        thread_id = thread_id or uuid.uuid4().hex
        final = self.graph.invoke(json_safe(initial_state(problem, plan)), self._config(thread_id))
        return Report.model_validate(final["report"])

    def resume(self, thread_id: str) -> Report:
        """Continue an interrupted run from its last checkpoint."""
        final = self.graph.invoke(None, self._config(thread_id))
        return Report.model_validate(final["report"])
