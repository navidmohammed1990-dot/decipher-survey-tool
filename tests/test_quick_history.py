"""Phase 9 — timing capture, call counts, and prompt size.

These pin the three things the performance brief asked about, so a future
change that reintroduces a second model call or an accumulating prompt fails
here rather than showing up as a slow conversion.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.classify.ollama import OllamaClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def ollama_body(response="{}", *, total=2.0, load=0.1, prompt_tokens=553,
                prompt_seconds=0.4, output_tokens=118, output_seconds=1.5):
    """A body shaped like a real non-streaming Ollama response."""
    ns = 1_000_000_000
    return {
        "model": "llama3",
        "response": response,
        "total_duration": int(total * ns),
        "load_duration": int(load * ns),
        "prompt_eval_count": prompt_tokens,
        "prompt_eval_duration": int(prompt_seconds * ns),
        "eval_count": output_tokens,
        "eval_duration": int(output_seconds * ns),
    }


class StubClient(OllamaClient):
    """A model that answers instantly and reports timings."""

    payload = {"element": "radio", "title_lines": [0], "confidence": 0.9, "notes": "ok"}

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.prompts: list[str] = []
        self.systems: list[str] = []

    def generate_json(self, system, prompt):
        self.systems.append(system)
        self.prompts.append(prompt)
        self.record_stats(ollama_body())
        return self.payload

    def status(self):
        return {"available": True, "reachable": True, "model_installed": True,
                "url": "stub", "model": "stub", "installed_models": ["stub"],
                "detail": "Ready."}


@pytest.fixture
def stub(monkeypatch):
    """Install one shared stub so its calls can be counted afterwards."""
    from app.api import routes_classify

    instance = StubClient()
    monkeypatch.setattr(routes_classify, "build_client", lambda: instance)
    return instance


# -- timings --------------------------------------------------------------


def test_timings_are_captured_from_the_response():
    client = OllamaClient()
    entry = client.record_stats(ollama_body())

    assert entry["total_seconds"] == 2.0
    assert entry["prompt_tokens"] == 553
    assert entry["output_tokens"] == 118
    assert client.stats == [entry]


def test_output_rate_is_computed():
    """Generation speed is what separates CPU from GPU inference."""
    entry = OllamaClient().record_stats(
        ollama_body(output_tokens=118, output_seconds=59.0)
    )
    assert entry["output_tokens_per_second"] == 2.0


def test_a_fast_rate_looks_like_gpu():
    entry = OllamaClient().record_stats(
        ollama_body(output_tokens=120, output_seconds=2.0)
    )
    assert entry["output_tokens_per_second"] == 60.0


def test_a_response_without_timings_is_tolerated():
    client = OllamaClient()
    assert client.record_stats({"response": "{}"}) is None
    assert client.stats == []


def test_zero_durations_do_not_divide_by_zero():
    entry = OllamaClient().record_stats(ollama_body(prompt_seconds=0, output_seconds=0))
    assert entry["output_tokens_per_second"] is None


def test_timings_reach_the_response(client, stub):
    body = client.post("/api/quick-convert", json={"text": "Q1. Pick one\nYes\nNo"}).json()

    assert len(body["timings"]) == 1
    assert body["timings"][0]["output_tokens"] == 118


# -- exactly one model call per question -----------------------------------


def test_one_question_costs_exactly_one_model_call(client, stub):
    """Checklist: calls per conversion must be one, not multiplied."""
    client.post("/api/quick-convert", json={"text": "Q1. Pick one\nYes\nNo"})
    assert len(stub.prompts) == 1


def test_five_questions_cost_five_calls(client, stub):
    """One per question, with no hidden second pass for tags or confidence."""
    text = "\n\n".join(f"Q{n}. Question {n}?" for n in range(1, 6))
    client.post("/api/quick-convert", json={"text": text})

    assert len(stub.prompts) == 5


def test_regenerating_costs_no_model_call(client, stub):
    body = client.post("/api/quick-convert", json={"text": "Q1. Pick one\nYes\nNo"}).json()
    before = len(stub.prompts)

    client.post("/api/quick-generate", json={"questions": body["questions"]})
    assert len(stub.prompts) == before


# -- prompt size -----------------------------------------------------------


def test_the_prompt_stays_small(client, stub):
    """Phase 7's instructions, a few lines, and a little precedent.

    The ceiling moved from 4000 to 6000 in Phase 17, deliberately: the prompt
    now carries the two most relevant known examples. What matters is that it
    is *fixed* - see the boundedness tests below - not that it is as small as
    it was before there was anything to carry.
    """
    client.post("/api/quick-convert", json={"text": "Q1. Pick one\nYes\nNo"})

    total = len(stub.systems[0]) + len(stub.prompts[0])
    assert total < 6000, f"prompt grew to {total} chars (~{total // 4} tokens)"


def test_the_prompt_does_not_grow_across_conversions(client, stub):
    """The paste path must not accumulate context as a session goes on."""
    for _ in range(4):
        client.post("/api/quick-convert", json={"text": "Q1. Pick one\nYes\nNo"})

    sizes = {len(system) for system in stub.systems}
    assert len(sizes) == 1, f"system prompt size varied across calls: {sizes}"


def test_quick_convert_carries_no_correction_history(client, stub):
    """Corrections are the one thing that could bloat a prompt over a session.

    Quick Convert opts out, so its prompt cannot grow with document review.
    """
    from app.classify.classifier import SYSTEM_PROMPT
    from app.classify.corrections import Correction, correction_memory
    from app.classify.seed_library import prompt_prefix as seed_prefix

    correction_memory.clear()
    correction_memory.use_document("some.docx")
    for number in range(3):
        correction_memory.record(Correction(
            label=f"Q{number}", original_lines=["x" * 400],
            ai_said={"element": "radio"}, sp_corrected_to={"element": "checkbox"},
        ))
    assert correction_memory.prompt_prefix() != ""

    client.post("/api/quick-convert", json={"text": "Q1. Pick one\nYes\nNo"})
    correction_memory.clear()

    system = stub.systems[0]
    assert "The survey programmer corrected an earlier question" not in system, (
        "document corrections must not reach the paste path"
    )
    # Seeded examples and dataset precedent are permanent and do travel with
    # every call; the document's own corrections are what must not.
    assert system.startswith(seed_prefix())
    assert system.endswith(SYSTEM_PROMPT)
    assert "x" * 400 not in system


# -- one tags round trip ---------------------------------------------------


def test_status_makes_a_single_tags_request(monkeypatch):
    import app.classify.ollama as module

    requests = []

    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"models": [{"name": "llama3:latest"}]}

    monkeypatch.setattr(module.httpx, "get", lambda url, **kw: (requests.append(url), Response())[1])
    OllamaClient(model="llama3").status()

    assert len(requests) == 1, "listing models already proves the runtime answers"
