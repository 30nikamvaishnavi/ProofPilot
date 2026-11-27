"""Shared helpers (plain module so tests also run under minimal runners)."""

from __future__ import annotations

from typing import Any


class FakeLLM:
    """Scripted LLM: returns the queued responses in order; raises if a response is an Exception."""

    name = "fake"

    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def complete_json(self, prompt: str) -> dict[str, Any]:
        self.prompts.append(prompt)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

