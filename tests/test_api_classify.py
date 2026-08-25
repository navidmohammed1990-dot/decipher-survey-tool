"""HTTP surface of Phase 2."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.parsing.docx_parser import parse_docx
from app.store import draft_store


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def parsed_payload(sample_docx):
    return parse_docx(sample_docx).model_dump(mode="json")


@pytest.fixture(autouse=True)
def _clean_store():
    draft_store.clear()
    yield
    draft_store.clear()


@pytest.fixture
def fake_ai(monkeypatch):
    """Patch the endpoint's client factory with a canned classifier."""
    from app.api import routes_classify
    from app.classify.ollama import OllamaClient

    def install(payload=None, error=None):
        class Fake(OllamaClient):
            def generate_json(self, system, prompt):
                if error:
                    raise error
                return payload

            def is_available(self):
                return error is None

        monkeypatch.setattr(routes_classify, "build_client", Fake)

    return install


CHECKBOX_ANSWER = {
    "element": "checkbox",
    "title_lines": [0],
    "comment_lines": [1],
    "option_lines": [2, 3, 4, 5],
    "confidence": 0.94,
    "notes": "select all that apply",
}


def test_classify_returns_questions(client, parsed_payload, fake_ai):
    fake_ai(payload=CHECKBOX_ANSWER)
    response = client.post("/api/classify", json=parsed_payload)

    assert response.status_code == 200
    body = response.json()
    assert [q["label"] for q in body["questions"]] == ["S1", "Q5", "Q6", "Q7"]
    assert body["summary"]["total"] == 4
    assert body["ai_available"] is True


def test_classify_preserves_formatting_from_phase_1(client, parsed_payload, fake_ai):
    fake_ai(payload=CHECKBOX_ANSWER)
    body = client.post("/api/classify", json=parsed_payload).json()

    q5 = next(q for q in body["questions"] if q["label"] == "Q5")
    assert [r["text"] for r in q5["title"] if r["bold"]] == ["purchased"]
    assert [o["raw_text"] for o in q5["options"]] == [
        "Brand A", "Brand B", "Brand C", "None of these",
    ]


def test_classify_stores_the_draft(client, parsed_payload, fake_ai):
    fake_ai(payload=CHECKBOX_ANSWER)
    client.post("/api/classify", json=parsed_payload)

    assert [q.label for q in draft_store.get().questions] == ["S1", "Q5", "Q6", "Q7"]


def test_threshold_query_param_controls_flagging(client, parsed_payload, fake_ai):
    # Options [2, 3] exist in every question of the fixture, so nothing is
    # flagged for a structural reason and confidence alone decides.
    fake_ai(payload={**CHECKBOX_ANSWER, "option_lines": [2, 3], "confidence": 0.8})

    lenient = client.post("/api/classify?threshold=0.5", json=parsed_payload).json()
    strict = client.post("/api/classify?threshold=0.95", json=parsed_payload).json()

    assert lenient["summary"]["flagged"] == 0
    assert strict["summary"]["flagged"] == 4
    assert strict["review_threshold"] == 0.95


def test_threshold_is_validated(client, parsed_payload, fake_ai):
    fake_ai(payload=CHECKBOX_ANSWER)
    assert client.post("/api/classify?threshold=1.5", json=parsed_payload).status_code == 422


def test_unreachable_ai_degrades_to_fallback_not_a_crash(client, parsed_payload, fake_ai):
    from app.classify.ollama import OllamaError

    fake_ai(error=OllamaError("connection refused"))
    response = client.post("/api/classify", json=parsed_payload)

    assert response.status_code == 200
    body = response.json()
    assert body["ai_available"] is False
    assert body["fallback_count"] == 4
    assert all(q["needs_review"] for q in body["questions"])
    assert all(q["confidence"] == 0.0 for q in body["questions"])
    # The warning must name the real cause and where to look for it.
    assert any("not reachable" in w for w in body["warnings"])
    assert any("fallback" in w for w in body["warnings"])
    assert "not reachable" in body["ai_detail"]


def test_ai_status_reports_reachability(client, fake_ai):
    from app.classify.ollama import OllamaError

    fake_ai(error=OllamaError("down"))
    assert client.get("/api/ai-status").json()["available"] is False


def test_classify_rejects_a_malformed_document(client):
    assert client.post("/api/classify", json={"blocks": "not a list"}).status_code == 422


def test_a_line_index_the_question_does_not_have_forces_review(client, parsed_payload, fake_ai):
    """Q7 has only 5 lines; an answer citing line 5 must not pass silently."""
    fake_ai(payload={**CHECKBOX_ANSWER, "confidence": 0.99})
    body = client.post("/api/classify?threshold=0.5", json=parsed_payload).json()

    q7 = next(q for q in body["questions"] if q["label"] == "Q7")
    assert q7["needs_review"] is True
    assert "unknown lines" in q7["ai_notes"]
