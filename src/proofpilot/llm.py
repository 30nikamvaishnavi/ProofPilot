"""Optional LLM access. Everything in ProofPilot works offline; the LLM only improves planning and repair."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Protocol


class LLMClient(Protocol):
    name: str

    def complete_json(self, prompt: str) -> dict[str, Any]: ...


class GeminiClient:
    """Thin wrapper over the google-genai SDK. Imported lazily so the package works without it."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash") -> None:
        from google import genai  # type: ignore[import-not-found]
        from google.genai import types  # type: ignore[import-not-found]

        self._types = types
        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.name = f"gemini:{model}"

    def complete_json(self, prompt: str) -> dict[str, Any]:
        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self._types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0),
        )
        return parse_json_object(resp.text or "")


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object even when the model wrapped it in markdown fences."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("model did not return a JSON object")
    return data


def get_llm() -> LLMClient | None:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        return None
    try:
        return GeminiClient(key, os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
    except Exception:  # SDK missing or misconfigured: fall back to offline mode
        return None
