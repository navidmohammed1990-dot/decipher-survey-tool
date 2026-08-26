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


class OllamaModelMissing(OllamaError):
    """The runtime is reachable but the configured model is not installed."""


def _tag_key(name: str) -> str:
    """Normalise a model name for comparison.

    Ollama resolves a bare name to its ``:latest`` tag, so a configured
    ``llama3`` and an installed ``llama3:latest`` are the same model.
    """
    name = name.strip()
    return name[: -len(":latest")] if name.endswith(":latest") else name


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
        self.stats: list[dict] = []
        """Per-call timings, newest last. See :meth:`record_stats`."""

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
        except httpx.HTTPError as exc:
            raise OllamaError(f"Could not reach Ollama at {self.base_url}: {exc}") from exc

        if response.status_code >= 400:
            raise self._error_for(response)

        try:
            body = response.json()
        except json.JSONDecodeError as exc:
            raise OllamaError("Ollama returned a non-JSON envelope.") from exc

        self.record_stats(body)

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

    def record_stats(self, body: dict) -> dict | None:
        """Keep the timings Ollama reports alongside every response.

        Ollama returns these on a non-streaming call and they answer the
        question `ollama ps` answers, without needing to catch a request in
        flight: generation speed on an 8B model is roughly 2-8 tokens/sec on
        CPU and 30-100+ on GPU, so eval_tokens_per_second separates the two.
        Durations are nanoseconds.
        """
        if not isinstance(body, dict) or "total_duration" not in body:
            return None

        def seconds(key: str) -> float:
            value = body.get(key) or 0
            return round(value / 1_000_000_000, 3)

        prompt_tokens = body.get("prompt_eval_count") or 0
        eval_tokens = body.get("eval_count") or 0
        prompt_seconds = seconds("prompt_eval_duration")
        eval_seconds = seconds("eval_duration")

        entry = {
            "model": body.get("model") or self.model,
            "total_seconds": seconds("total_duration"),
            "load_seconds": seconds("load_duration"),
            "prompt_tokens": prompt_tokens,
            "prompt_seconds": prompt_seconds,
            "output_tokens": eval_tokens,
            "output_seconds": eval_seconds,
            "prompt_tokens_per_second": round(prompt_tokens / prompt_seconds, 1) if prompt_seconds else None,
            "output_tokens_per_second": round(eval_tokens / eval_seconds, 1) if eval_seconds else None,
        }
        self.stats.append(entry)
        logger.info(
            "Ollama %s: %.1fs total (load %.1fs, prompt %d tok in %.1fs, "
            "output %d tok in %.1fs, %s tok/s output)",
            entry["model"], entry["total_seconds"], entry["load_seconds"],
            entry["prompt_tokens"], entry["prompt_seconds"],
            entry["output_tokens"], entry["output_seconds"],
            entry["output_tokens_per_second"],
        )
        return entry

    def _error_for(self, response) -> OllamaError:
        """Turn an HTTP error into one that says what actually went wrong.

        Ollama answers /api/generate with 404 when the *model* is unknown, not
        only when the URL is. Reporting the bare status sends people hunting for
        a URL problem, so the body — which names the real cause — is surfaced.
        """
        detail = ""
        try:
            detail = str(response.json().get("error") or "").strip()
        except (json.JSONDecodeError, AttributeError, ValueError):
            detail = response.text.strip()[:200]

        if response.status_code == 404 and "not found" in detail.lower():
            installed = self.list_models()
            available = f" Installed models: {', '.join(installed)}." if installed else ""
            return OllamaModelMissing(
                f"Ollama is running at {self.base_url} but the model "
                f"'{self.model}' is not installed.{available} "
                f"Pull it with `ollama pull {self.model}`, or point the tool at a "
                f"model you already have by setting DECIPHER_OLLAMA_MODEL. "
                f"(Ollama said: {detail})"
            )

        suffix = f": {detail}" if detail else ""
        return OllamaError(
            f"Ollama returned HTTP {response.status_code} from "
            f"{self.base_url}/api/generate{suffix}"
        )

    def list_models(self) -> list[str]:
        """Model names the runtime currently has installed."""
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            response.raise_for_status()
            models = response.json().get("models") or []
        except (httpx.HTTPError, json.JSONDecodeError, AttributeError):
            return []
        return [
            str(entry.get("name") or entry.get("model") or "").strip()
            for entry in models
            if entry.get("name") or entry.get("model")
        ]

    def is_reachable(self) -> bool:
        """Whether the runtime answers at all, regardless of which models it has."""
        try:
            return httpx.get(f"{self.base_url}/api/tags", timeout=3.0).status_code == 200
        except httpx.HTTPError:
            return False

    def has_model(self, installed: list[str] | None = None) -> bool:
        installed = self.list_models() if installed is None else installed
        wanted = _tag_key(self.model)
        return any(_tag_key(name) == wanted for name in installed)

    def is_available(self) -> bool:
        """Ready to classify: reachable *and* holding the configured model.

        Checking only reachability reported "AI ready" while every call 404'd.
        """
        installed = self.list_models()
        return bool(installed) and self.has_model(installed)

    def status(self) -> dict:
        """A full picture for the UI, so a misconfiguration is visible up front.

        One /api/tags round trip: listing the models already proves the runtime
        answers, so probing reachability separately was a wasted request.
        """
        installed = self.list_models()
        reachable = bool(installed) or self.is_reachable()
        model_present = self.has_model(installed) if installed else False

        if not reachable:
            detail = f"Ollama is not reachable at {self.base_url}."
        elif not installed:
            detail = f"Ollama is running at {self.base_url} but has no models installed."
        elif not model_present:
            detail = (
                f"Ollama is running but '{self.model}' is not installed. "
                f"Installed: {', '.join(installed)}. Set DECIPHER_OLLAMA_MODEL to one "
                f"of those, or run `ollama pull {self.model}`."
            )
        else:
            detail = f"Ready with {self.model}."

        return {
            "available": reachable and model_present,
            "reachable": reachable,
            "model_installed": model_present,
            "url": self.base_url,
            "model": self.model,
            "installed_models": installed,
            "detail": detail,
        }
