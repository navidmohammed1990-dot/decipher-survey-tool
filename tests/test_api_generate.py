"""HTTP surface of Phase 3."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.survey import OptionLine, Question, QuestionDraft
from app.models.document import TextRun
from app.store import draft_store


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _clean_store():
    draft_store.clear()
    yield
    draft_store.clear()


def sample_questions():
    return [
        Question(
            label="Q5",
            element="checkbox",
            title=[TextRun(text="Which "), TextRun(text="brands", bold=True), TextRun(text="?")],
            comment=[TextRun(text="Select all that apply.")],
            options=[OptionLine(raw_text=t) for t in
                     ["Brand A", "Brand B", "Other (please specify)", "None of these"]],
            confidence=0.94,
            needs_review=False,
        ),
        Question(
            label="Q6",
            element="radio",
            title=[TextRun(text="Pick one")],
            options=[OptionLine(raw_text="Yes"), OptionLine(raw_text="No")],
            confidence=0.9,
            needs_review=False,
        ),
    ]


def seed_draft(questions=None):
    draft_store.replace(
        QuestionDraft(questions=questions or sample_questions(), source_filename="brand.docx")
    )


def payload(questions=None, **kwargs):
    return {"questions": [q.model_dump(mode="json") for q in (questions or sample_questions())],
            **kwargs}


def test_generate_returns_xml(client):
    body = client.post("/api/generate", json=payload()).json()

    assert body["question_count"] == 2
    assert body["well_formed"] is True
    assert '<checkbox label="Q5"' in body["xml"]
    assert body["xml"].count("<suspend/>") == 2


def test_generate_applies_the_r91_r99_convention(client):
    xml_text = client.post("/api/generate", json=payload()).json()["xml"]

    assert '<row label="r91" open="1" openSize="25" randomize="0">' in xml_text
    assert '<row label="r99" randomize="0" exclusive="1">None of these</row>' in xml_text


def test_generate_preserves_formatting(client):
    xml_text = client.post("/api/generate", json=payload()).json()["xml"]
    assert "<title>Which <b>brands</b>?</title>" in xml_text


def test_generate_is_deterministic_across_requests(client):
    first = client.post("/api/generate", json=payload()).json()["xml"]
    for _ in range(5):
        assert client.post("/api/generate", json=payload()).json()["xml"] == first


def test_generate_falls_back_to_the_stored_draft(client):
    seed_draft()
    body = client.post("/api/generate", json={}).json()

    assert body["question_count"] == 2
    assert '<checkbox label="Q5"' in body["xml"]


def test_generate_with_nothing_to_do_is_a_clear_error(client):
    response = client.post("/api/generate", json={})
    assert response.status_code == 400
    assert "Classify a document first" in response.json()["detail"]


def test_generate_rejects_an_unsupported_element(client):
    bad = [Question(label="Q1", element="dropdown")]
    response = client.post("/api/generate", json=payload(bad))

    assert response.status_code == 422
    assert "not a supported element" in response.json()["detail"]


def test_generate_warns_about_still_flagged_questions(client):
    flagged = sample_questions()
    flagged[0].needs_review = True
    body = client.post("/api/generate", json=payload(flagged)).json()

    assert any("Q5 is still flagged" in w for w in body["warnings"])
    assert body["xml"], "a warning must not block generation"


def test_wrap_option_produces_standalone_parseable_xml(client):
    body = client.post("/api/generate", json=payload(wrap=True)).json()

    root = ET.fromstring(body["xml"])
    assert root.tag == "survey"
    assert len(root.findall("checkbox")) == 1


def test_unwrapped_fragments_declare_no_namespaces(client):
    xml_text = client.post("/api/generate", json=payload()).json()["xml"]
    assert "xmlns:" not in xml_text


def test_single_question_preview(client):
    seed_draft()
    body = client.post("/api/generate/Q6").json()

    assert body["question_count"] == 1
    assert '<radio label="Q6"' in body["xml"]
    assert "Q5" not in body["xml"]


def test_single_question_preview_needs_a_known_label(client):
    seed_draft()
    assert client.post("/api/generate/Q99").status_code == 404


def test_export_downloads_a_wrapped_file(client):
    seed_draft()
    response = client.get("/api/export.xml")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/xml")
    assert 'filename="brand_base.xml"' in response.headers["content-disposition"]
    assert ET.fromstring(response.text).tag == "survey"


def test_export_can_return_bare_fragments(client):
    seed_draft()
    text = client.get("/api/export.xml?wrap=false").text

    assert "<survey" not in text
    assert '<checkbox label="Q5"' in text


def test_export_without_a_draft_is_a_clear_error(client):
    assert client.get("/api/export.xml").status_code == 400
