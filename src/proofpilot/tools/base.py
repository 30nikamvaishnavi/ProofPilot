"""Tool abstraction: every tool pairs a solver with an *independent* verifier."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import BaseModel

from ..schemas import VerifyResult


@dataclass
class ToolOutput:
    """JSON-serialisable value plus a human-readable rendering."""

    value: Any
    display: str


class Tool(ABC):
    name: ClassVar[str]
    domain: ClassVar[str]  # "math" | "cs" | "finance"
    description: ClassVar[str]
    arg_hint: ClassVar[str]  # shown to the LLM planner
    assumptions: ClassVar[list[str]] = []
    ArgsModel: ClassVar[type[BaseModel]]

    def validate(self, args: dict[str, Any]) -> BaseModel:
        return self.ArgsModel(**args)

    @abstractmethod
    def run(self, args: Any) -> ToolOutput:
        """Compute the answer with the primary method."""

    @abstractmethod
    def verify(self, args: Any, output: ToolOutput) -> VerifyResult:
        """Re-check ``output`` with a different method. Must read only ``output.value``."""

    def repair_args(self, args: dict[str, Any], attempt: int, error: str | None) -> dict[str, Any] | None:
        """Deterministic repair. Return None when nothing can be changed."""
        fixed = {k: _sanitize(v) for k, v in args.items()}
        return fixed if fixed != args else None


_REPLACEMENTS = (("×", "*"), ("÷", "/"), ("−", "-"), ("≤", "<="), ("≥", ">="), ("²", "**2"), ("³", "**3"))


def _sanitize(value: Any) -> Any:
    if isinstance(value, str):
        out = value.strip().rstrip(".,;?").strip()
        for a, b in _REPLACEMENTS:
            out = out.replace(a, b)
        out = re.sub(r"\^", "**", out)
        return out
    return value


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return self._tools[name]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def catalog(self) -> list[dict[str, str]]:
        return [
            {"name": t.name, "domain": t.domain, "description": t.description, "args": t.arg_hint}
            for t in self._tools.values()
        ]


def default_registry() -> ToolRegistry:
    from .cs_tools import MinimumSpanningTree, ShortestPath, TopologicalSort
    from .finance_tools import BlackScholes, PortfolioVariance
    from .math_tools import (
        Differentiate,
        Integrate,
        LinearSystem,
        MatrixOps,
        NumberTheory,
        SolveEquation,
    )

    reg = ToolRegistry()
    for cls in (
        SolveEquation,
        Differentiate,
        Integrate,
        MatrixOps,
        LinearSystem,
        NumberTheory,
        ShortestPath,
        MinimumSpanningTree,
        TopologicalSort,
        BlackScholes,
        PortfolioVariance,
    ):
        reg.register(cls())
    return reg
