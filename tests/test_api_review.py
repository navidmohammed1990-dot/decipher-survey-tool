"""Phase 4 — the survey programmer's review flow."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.document import TextRun
from app.models.survey import OptionLine, Question, QuestionDraft
from app.store import draft_store


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def seeded():
    draft_store.replace(QuestionDraft(
        source_filename="brand.docx",
        questions=[
            Question(
                label="Q5", element="checkbox",
                title=[TextRun(text="Which "), TextRun(text="brands", bold=True)],
                options=[OptionLine(raw_text="Brand A"), OptionLine(raw_text="Brand B")],
                confidence=0.94, needs_review=False,
            ),
            Question(
                label="Q6", element="radio",
                title=[TextRun(text="Uncertain question")],
                options=[OptionLine(raw_text="Yes")],
                confidence=0.3, needs_review=True, ai_notes="low confidence",
            ),
        ],
    ))
    yield
    draft_store.clear()


def patch(client, label, **fields):
    return client.patch(f"/api/questions/{label}", json=fields)


def test_get_questions_returns_the_draft(client):
    body = client.get("/api/questions").json()

    assert [q["label"] for q in body["questions"]] == ["Q5", "Q6"]
    assert body["source_filename"] == "brand.docx"
    assert body["summary"] == {"total": 2, "flagged": 1, "confident": 1}


def test_element_can_be_changed(client):
    body = patch(client, "Q5", element="radio").json()

    assert body["element"] == "radio"
    assert draft_store.get().find("Q5").element == "radio"


def test_a_high_confidence_question_is_still_editable(client):
    """Confidence is a hint about where to look, never a lock."""
    assert patch(client, "Q5", element="select").status_code == 200
    assert patch(client, "Q5", title="Rewritten by the programmer").status_code == 200


def test_options_are_edited_as_line_separated_text(client):
    body = patch(client, "Q5", options="Brand A\nBrand B\nBrand C\n\n  \n").json()

    assert [o["raw_text"] for o in body["options"]] == ["Brand A", "Brand B", "Brand C"]


def test_grid_rows_and_cols_are_editable(client):
    body = patch(client, "Q6", element="radio_grid", rows="R1\nR2", cols="Agree\nDisagree").json()

    assert [r["raw_text"] for r in body["rows"]] == ["R1", "R2"]
    assert [c["raw_text"] for c in body["cols"]] == ["Agree", "Disagree"]


def test_title_and_comment_are_editable(client):
    body = patch(client, "Q6", title="New title", comment="New instruction").json()

    assert body["title"][0]["text"] == "New title"
    assert body["comment"][0]["text"] == "New instruction"


def test_clearing_a_field_empties_it(client):
    assert patch(client, "Q5", comment="").json()["comment"] == []


def test_dev_notes_are_stored_but_never_exported(client):
    patch(client, "Q5", dev_notes="Check routing with the client")
    assert draft_store.get().find("Q5").dev_notes == "Check routing with the client"

    xml_text = client.post("/api/generate", json={}).json()["xml"]
    assert "Check routing" not in xml_text


def test_editing_clears_the_review_flag(client):
    """A programmer edit is a decision, so it answers the request for review."""
    assert draft_store.get().find("Q6").needs_review is True
    assert patch(client, "Q6", element="radio").json()["needs_review"] is False


def test_dev_notes_alone_do_not_clear_the_review_flag(client):
    """Jotting a note is not the same as resolving the question."""
    assert patch(client, "Q6", dev_notes="come back to this").json()["needs_review"] is True


def test_review_flag_can_be_set_explicitly(client):
    assert patch(client, "Q5", needs_review=True).json()["needs_review"] is True
    assert client.get("/api/questions").json()["summary"]["flagged"] == 2


def test_a_partial_patch_leaves_other_fields_alone(client):
    patch(client, "Q5", dev_notes="note only")
    question = draft_store.get().find("Q5")

    assert question.element == "checkbox"
    assert [o.raw_text for o in question.options] == ["Brand A", "Brand B"]
    assert question.title[1].bold is True, "imported formatting survived an unrelated edit"


def test_editing_a_title_replaces_its_formatting(client):
    """Retyped text is plain: reapplying old bold to new words would be worse."""
    body = patch(client, "Q5", title="Plain replacement").json()
    assert body["title"] == [{"text": "Plain replacement", "bold": False,
                              "italic": False, "underline": False,
                              "strike": False, "color": None}]


def test_unknown_element_is_rejected_with_a_helpful_message(client):
    response = patch(client, "Q5", element="dropdown")

    assert response.status_code == 422
    assert "not supported" in response.text


def test_unknown_label_is_a_404(client):
    assert patch(client, "Q99", element="radio").status_code == 404


def test_edits_flow_through_to_generated_xml(client):
    patch(client, "Q5", element="radio", options="Yes\nNo\nNone of these")
    xml_text = client.post("/api/generate", json={}).json()["xml"]

    assert '<radio label="Q5"' in xml_text
    assert '<row label="r1" value="1">Yes</row>' in xml_text
    assert '<row label="r99" value="99" randomize="0">None of these</row>' in xml_text
    assert "exclusive" not in xml_text, "radio does not need exclusive"


def test_draft_can_be_cleared(client):
    assert client.delete("/api/questions").status_code == 204
    assert client.get("/api/questions").json()["questions"] == []
