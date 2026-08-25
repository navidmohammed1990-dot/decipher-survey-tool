"""The brief's testing checklist, as executable tests.

Round-trips a real questionnaire DOCX through parse -> classify -> review ->
generate, and pins the guarantees the whole pipeline is supposed to offer.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.store import draft_store


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _clean():
    draft_store.clear()
    yield
    draft_store.clear()


@pytest.fixture
def offline_ai(monkeypatch):
    """Force the fallback path by pointing the classifier at a dead port."""
    from app.api import routes_classify
    from app.classify.ollama import OllamaClient

    monkeypatch.setattr(
        routes_classify, "build_client",
        lambda: OllamaClient(base_url="http://127.0.0.1:1", timeout=1.0),
    )


@pytest.fixture
def scripted_ai(monkeypatch):
    """A model that answers each fixture question correctly, by label."""
    from app.api import routes_classify
    from app.classify.ollama import OllamaClient

    answers = {
        "S1": {"element": "radio", "title_lines": [0], "comment_lines": [1],
               "option_lines": [2, 3, 4, 5], "confidence": 0.96, "notes": "select one"},
        "Q5": {"element": "checkbox", "title_lines": [0], "comment_lines": [1],
               "option_lines": [2, 3, 4, 5], "confidence": 0.94, "notes": "select all"},
        "Q6": {"element": "radio_grid", "title_lines": [0], "comment_lines": [1],
               "row_lines": [5, 6], "col_lines": [3, 4], "confidence": 0.91, "notes": "grid"},
        "Q7": {"element": "textarea", "title_lines": [0],
               "confidence": 0.6, "notes": "open ended, unsure"},
    }

    class Scripted(OllamaClient):
        def generate_json(self, system, prompt):
            label = prompt.split("\n", 1)[0].removeprefix("Question label: ").strip()
            return answers[label]

        def is_available(self):
            return True

    monkeypatch.setattr(routes_classify, "build_client", Scripted)


def parse(client, sample_docx):
    response = client.post(
        "/api/parse",
        files={"file": ("brand_tracker.docx", sample_docx.read_bytes(),
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    )
    assert response.status_code == 200
    return response.json()


# -- checklist: full round trip -------------------------------------------


def test_parse_classify_review_generate(client, sample_docx, scripted_ai):
    parsed = parse(client, sample_docx)
    assert [q["label"] for q in parsed["questions"] if not q["is_preamble"]] == [
        "S1", "Q5", "Q6", "Q7",
    ]

    classified = client.post("/api/classify", json=parsed).json()
    assert classified["ai_available"] is True
    assert [q["element"] for q in classified["questions"]] == [
        "radio", "checkbox", "radio_grid", "textarea",
    ]
    # Q7 came back at 0.6, below the 0.75 default.
    assert classified["summary"]["flagged"] == 1

    # The programmer corrects the one the model was unsure about.
    corrected = client.patch("/api/questions/Q7", json={"element": "text"}).json()
    assert corrected["element"] == "text"
    assert corrected["needs_review"] is False
    assert client.get("/api/questions").json()["summary"]["flagged"] == 0

    generated = client.post("/api/generate", json={}).json()
    assert generated["question_count"] == 4
    assert generated["well_formed"] is True
    assert generated["warnings"] == []
    assert generated["xml"].count("<suspend/>") == 4


def test_the_docx_formatting_reaches_the_final_xml(client, sample_docx, scripted_ai):
    """Bold and italic set in Word survive parse, classify and generate."""
    client.post("/api/classify", json=parse(client, sample_docx))
    xml_text = client.post("/api/generate", json={}).json()["xml"]

    assert "<b>purchased</b>" in xml_text
    assert "<i>6 months</i>" in xml_text


# -- checklist: checkbox matches the canonical template --------------------


def test_generated_checkbox_matches_the_canonical_structure(client, sample_docx, scripted_ai):
    client.post("/api/classify", json=parse(client, sample_docx))
    xml_text = client.post("/api/generate/Q5").json()["xml"]

    assert 'atleast="1"' in xml_text
    assert 'uses="atm1d.10"' in xml_text
    assert 'ss:listDisplay="1"' in xml_text
    assert 'fwidth="1000"' in xml_text
    assert "<comment>" in xml_text
    assert '<row label="r99" randomize="0" exclusive="1">None of these</row>' in xml_text
    assert "<validate>CheckBlank(1,Q5)</validate>" in xml_text
    assert "value=" not in xml_text, "checkbox rows omit value"


def test_generated_grid_has_rows_and_columns(client, sample_docx, scripted_ai):
    client.post("/api/classify", json=parse(client, sample_docx))
    xml_text = client.post("/api/generate/Q6").json()["xml"]

    assert '<row label="r1">The brand is good value</row>' in xml_text
    assert '<col label="c1"><b>Agree</b></col>' in xml_text


# -- checklist: well-formedness -------------------------------------------


def test_fragments_are_well_formed_inside_a_survey_root(client, sample_docx, scripted_ai):
    client.post("/api/classify", json=parse(client, sample_docx))

    bare = client.post("/api/generate", json={}).json()
    assert "xmlns:" not in bare["xml"], "namespaces belong on the root, not per fragment"

    wrapped = client.post("/api/generate", json={"wrap": True}).json()
    root = ET.fromstring(wrapped["xml"])
    assert root.tag == "survey"


def test_the_downloaded_export_parses(client, sample_docx, scripted_ai):
    client.post("/api/classify", json=parse(client, sample_docx))
    response = client.get("/api/export.xml")

    assert response.status_code == 200
    assert ET.fromstring(response.text).tag == "survey"
    assert 'filename="brand_tracker_base.xml"' in response.headers["content-disposition"]


# -- checklist: determinism ------------------------------------------------


def test_identical_input_always_produces_identical_output(client, sample_docx, scripted_ai):
    client.post("/api/classify", json=parse(client, sample_docx))
    draft = client.get("/api/questions").json()
    request = {"questions": draft["questions"]}

    outputs = {client.post("/api/generate", json=request).json()["xml"] for _ in range(10)}
    assert len(outputs) == 1


def test_reclassifying_the_same_document_is_stable(client, sample_docx, scripted_ai):
    parsed = parse(client, sample_docx)
    first = client.post("/api/classify", json=parsed).json()["questions"]
    second = client.post("/api/classify", json=parsed).json()["questions"]
    assert first == second


# -- checklist: degradation, not crashes ----------------------------------


def test_an_unreachable_model_degrades_to_the_fallback(client, sample_docx, offline_ai):
    parsed = parse(client, sample_docx)
    classified = client.post("/api/classify", json=parsed)

    assert classified.status_code == 200, "an offline model must not 500"
    body = classified.json()
    assert body["ai_available"] is False
    assert body["fallback_count"] == len(body["questions"])
    assert all(q["needs_review"] for q in body["questions"])
    assert all(q["element"] == "radio" for q in body["questions"])
    assert all(q["confidence"] == 0.0 for q in body["questions"])


def test_the_fallback_still_produces_usable_well_formed_xml(client, sample_docx, offline_ai):
    """Degraded is not broken: the SP still gets something to correct."""
    client.post("/api/classify", json=parse(client, sample_docx))
    generated = client.post("/api/generate", json={}).json()

    assert generated["well_formed"] is True
    assert generated["question_count"] == 4
    assert len(generated["warnings"]) == 4, "every fallback question warns it needs review"
    assert ET.fromstring(client.get("/api/export.xml").text).tag == "survey"
