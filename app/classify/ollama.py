"""Thin client for a local Ollama runtime.

Kept deliberately small and free of survey logic: everything above it treats a
classification failure and an unreachable server identically, which is what
makes the fallback path easy to exercise in tests.
"""

from __future__ import annotations

import json
import logging

import httpx

logger = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    """Raised for any failure to obtain usable JSON from the model."""


class OllamaClient:
    """Calls ``/api/generate`` and returns the parsed JSON response."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1",
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate_json(self, system: str, prompt: str) -> dict:
        """Ask the model for one JSON object.

        ``format="json"`` and ``temperature=0`` are both required by the brief:
        the first constrains the model to emit JSON, the second keeps repeated
        runs over the same questionnaire stable.
        """
        payload = {
            "model": self.model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }

        try:
            response = httpx.post(
                f"{self.base_url}/api/generate", json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPError as exc:
            raise OllamaError(f"Ollama request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise OllamaError("Ollama returned a non-JSON envelope.") from exc

        raw = body.get("response")
        if not isinstance(raw, str) or not raw.strip():
            raise OllamaError("Ollama returned an empty response.")

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OllamaError(f"Model output was not valid JSON: {raw[:200]!r}") from exc

        if not isinstance(parsed, dict):
            raise OllamaError(f"Expected a JSON object, got {type(parsed).__name__}.")
        return parsed

    def is_available(self) -> bool:
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=3.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False
