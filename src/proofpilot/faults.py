"""Fault injection used by the evaluation to measure what the verifier buys you."""

from __future__ import annotations

import random
from typing import Any

from .tools.base import ToolOutput


def corrupt_value(value: Any) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value * 1.1 + 1
    if isinstance(value, str):
        return f"({value}) * 2 + 1"  # also wrong for antiderivatives (a bare "+ 1" would still be valid)
    if isinstance(value, list) and value:
        return [corrupt_value(value[0]), *value[1:]]
    if isinstance(value, dict) and value:
        first = next(iter(value))
        return {**value, first: corrupt_value(value[first])}
    return value


def make_fault_hook(rate: float, seed: int = 0):
    """Return a hook that corrupts a tool output with probability ``rate`` (seeded, reproducible)."""
    rng = random.Random(seed)

    def hook(tool: str, out: ToolOutput) -> ToolOutput:
        if rng.random() < rate:
            return ToolOutput(corrupt_value(out.value), out.display + "  <corrupted>")
        return out

    return hook
