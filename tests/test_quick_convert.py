"""Phase 8 — Quick Convert.

A second entry point over the same engine. The load-bearing test is
:func:`test_pasted_text_and_docx_generate_identical_xml`: if the two paths ever
diverge, the reuse promise is broken and this file should fail loudly.
"""

from __future__ import annotations

import time

import docx
import pytest
from fastapi.testclient import TestClient

from app.classify.paste import join_cells, normalise_lines, split_questions
from app.main import app
from app.store import draft_store


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def scripted_ai(monkeypatch):
    """A model that answers each pasted block plausibly, keyed by label."""
    from app.api import routes_classify
    from app.classify.ollama import OllamaClient

    answers = {
        "Q1": {"element": "checkbox", "title_lines": [0], "comment_lines": [1],
               "option_lines": [2, 3, 4], "confidence": 0.94, "notes": "select all"},
        "Q2": {"element": "radio", "title_lines": [0], "option_lines": [1, 2, 3],
               "routing_lines": [4], "confidence": 0.95, "notes": "gender, SC; line 4 is a quota note"},
        "Q3": {"element": "number", "title_lines": [0], "confidence": 0.9, "notes": "numeric entry"},
        "Q4": {"element": "textarea", "title_lines": [0], "confidence": 0.88, "notes": "open end"},
        "Q5": {"element": "radio", "title_lines": [0], "option_lines": [1, 2],
               "confidence": 0.92, "notes": "yes/no"},
    }

    class Scripted(OllamaClient):
        def generate_json(self, system, prompt):
            label = prompt.split("\n", 1)[0].removeprefix("Question label: ").strip()
            return answers.get(label, {"element": "text", "title_lines": [0],
                                       "confidence": 0.4, "notes": "unsure"})

        def is_available(self):
            return True

        def status(self):
            return {"available": True, "reachable": True, "model_installed": True,
                    "url": "stub", "model": "stub", "installed_models": ["stub"],
                    "detail": "Ready with stub."}

    monkeypatch.setattr(routes_classify, "build_client", Scripted)


SINGLE = """Q1. Which of these have you bought?
Please select all that apply.
Brand A
Brand B
Other, please specify
"""

MULTI = """Q1. Which of these have you bought?
Please select all that apply.
Brand A
Brand B
Other, please specify

Q2. What is your gender?
Male\t1
Female\t2
Other\t97
RANDOMLY ASSIGN OTHER INTO MALE/FEMALE QUOTAS

Q3. How old are you?

Q4. Why did you choose that brand?

Q5. Do you hold a licence?
Yes
No
"""


def convert(client, text, **kwargs):
    return client.post("/api/quick-convert", json={"text": text, **kwargs})


# -- splitting -------------------------------------------------------------


def test_a_copied_table_row_stays_one_line():
    assert join_cells("Male\t1") == "Male | 1"
    assert join_cells("Other\t97") == "Other | 97"


def test_a_line_without_cells_is_left_alone():
    assert join_cells("Please select all that apply.") == "Please select all that apply."


def test_blank_lines_are_dropped():
    assert normalise_lines("a\n\n\n b \n") == ["a", "b"]


def test_windows_line_endings_are_handled():
    assert normalise_lines("a\r\nb\r\n") == ["a", "b"]


def test_a_single_question_splits_into_one_block():
    blocks, warnings = split_questions(SINGLE)

    assert len(blocks) == 1
    assert blocks[0].label == "Q1"
    assert [line.text for line in blocks[0].lines] == [
        "Which of these have you bought?", "Please select all that apply.",
        "Brand A", "Brand B", "Other, please specify",
    ]
    assert warnings == []


def test_the_label_is_stripped_from_the_first_line():
    blocks, _ = split_questions("Q7. What is your postcode?")
    assert blocks[0].lines[0].text == "What is your postcode?"


def test_several_questions_split_on_their_labels():
    blocks, _ = split_questions(MULTI)
    assert [b.label for b in blocks] == ["Q1", "Q2", "Q3", "Q4", "Q5"]


def test_pasted_table_codes_survive_the_split():
    blocks, _ = split_questions(MULTI)
    q2 = next(b for b in blocks if b.label == "Q2")

    assert [line.text for line in q2.lines[1:4]] == ["Male | 1", "Female | 2", "Other | 97"]
    assert q2.lines[1].features.trailing_numeric_code == "1"
    assert q2.lines[3].features.trailing_numeric_code == "97"


def test_a_routing_line_is_hinted_but_not_pre_decided():
    blocks, _ = split_questions(MULTI)
    q2 = next(b for b in blocks if b.label == "Q2")

    assert q2.lines[4].features.matches_routing_keyword is True
    assert q2.lines[4].text.startswith("RANDOMLY ASSIGN")


def test_typed_option_numbers_become_codes():
    blocks, _ = split_questions("Q1. Pick one\n1. Yes\n2. No\n97. Other, please specify")
    texts = [line.text for line in blocks[0].lines]

    assert texts == ["Pick one", "Yes", "No", "Other, please specify"]
    assert [line.literal_marker for line in blocks[0].lines[1:]] == ["1.", "2.", "97."]


def test_an_unlabelled_paste_becomes_one_question():
    """A programmer who selected one question should not be told off for it."""
    blocks, warnings = split_questions("Which of these have you bought?\nBrand A\nBrand B")

    assert len(blocks) == 1
    assert blocks[0].label == "Q1"
    assert blocks[0].synthesised_label is True
    assert any("treated the whole paste as one question" in w for w in warnings)


def test_plain_numbering_is_only_a_fallback():
    blocks, warnings = split_questions("1. First question\nYes\nNo\n2. Second question\nYes\nNo")

    assert [b.label for b in blocks] == ["1", "2"]
    assert any("plain numbering" in w for w in warnings)


def test_numbered_options_inside_a_labelled_question_are_not_a_split():
    blocks, _ = split_questions("Q1. Pick one\n1. Yes\n2. No")
    assert len(blocks) == 1, "the Q label wins; 1./2. are its options"


def test_empty_text_yields_nothing():
    blocks, warnings = split_questions("   \n\n ")
    assert blocks == []
    assert warnings


# -- the endpoint ----------------------------------------------------------


def test_a_single_question_converts(client, scripted_ai):
    body = convert(client, SINGLE).json()

    assert len(body["questions"]) == 1
    question = body["questions"][0]
    assert question["label"] == "Q1"
    assert question["element"] == "checkbox"
    assert [o["raw_text"] for o in question["options"]] == [
        "Brand A", "Brand B", "Other, please specify",
    ]
    assert body["well_formed"] is True
    assert '<checkbox label="Q1"' in body["xml"]


def test_five_questions_convert_together(client, scripted_ai):
    body = convert(client, MULTI).json()

    assert [q["label"] for q in body["questions"]] == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    assert [q["element"] for q in body["questions"]] == [
        "checkbox", "radio", "number", "textarea", "radio",
    ]
    assert body["xml"].count("<suspend/>") == 5


def test_pasted_codes_reach_the_generated_xml(client, scripted_ai):
    xml_text = convert(client, MULTI).json()["xml"]

    assert '<row label="r1" value="1">Male</row>' in xml_text
    assert '<row label="r2" value="2">Female</row>' in xml_text
    assert '<row label="r97" value="97">Other</row>' in xml_text


def test_routing_notes_are_separate_and_never_exported(client, scripted_ai):
    body = convert(client, MULTI).json()
    q2 = next(q for q in body["questions"] if q["label"] == "Q2")

    assert q2["routing_notes"] == ["RANDOMLY ASSIGN OTHER INTO MALE/FEMALE QUOTAS"]
    assert "RANDOMLY ASSIGN" not in body["xml"]


def test_the_r91_convention_still_applies(client, scripted_ai):
    xml_text = convert(client, SINGLE).json()["xml"]
    assert '<row label="r91" open="1" openSize="25" randomize="0">Other, please specify</row>' in xml_text


def test_an_empty_paste_is_a_clear_error(client, scripted_ai):
    response = convert(client, "   ")
    assert response.status_code == 400
    assert "paste some text" in response.json()["detail"]


def test_an_oversized_paste_suggests_the_document_flow(client, scripted_ai):
    response = convert(client, "Q1. x\n" + ("option line\n" * 20000))
    assert response.status_code == 413
    assert "document upload" in response.json()["detail"]


def test_conversion_is_fast_for_five_questions(client, scripted_ai):
    """A few questions should feel instant; no batching machinery needed."""
    started = time.monotonic()
    convert(client, MULTI)
    assert time.monotonic() - started < 5.0


# -- editing and re-generating --------------------------------------------


def test_editing_a_field_regenerates_without_reclassifying(client, scripted_ai):
    body = convert(client, SINGLE).json()
    questions = body["questions"]
    questions[0]["element"] = "radio"

    regenerated = client.post("/api/quick-generate", json={"questions": questions}).json()

    assert '<radio label="Q1"' in regenerated["xml"]
    assert regenerated["well_formed"] is True


def test_regenerating_needs_no_model_call(client, monkeypatch, scripted_ai):
    body = convert(client, SINGLE).json()

    from app.api import routes_classify

    def explode():
        raise AssertionError("re-generating must not call the model")

    monkeypatch.setattr(routes_classify, "build_client", explode)
    assert client.post("/api/quick-generate", json={"questions": body["questions"]}).status_code == 200


def test_regenerating_with_nothing_is_a_clear_error(client):
    assert client.post("/api/quick-generate", json={"questions": []}).status_code == 400


def test_an_unsupported_element_is_rejected(client):
    response = client.post("/api/quick-generate", json={
        "questions": [{"label": "Q1", "element": "dropdown"}]
    })
    assert response.status_code == 422


# -- the engine really is shared ------------------------------------------


def test_pasted_text_and_docx_generate_identical_xml(client, scripted_ai, tmp_path):
    """The same question, pasted and uploaded, must produce the same bytes.

    This is the whole reuse promise. If it fails, Quick Convert has grown its
    own engine.
    """
    text = "Q1. Which of these have you bought?\nPlease select all that apply.\nBrand A\nBrand B\nOther, please specify"

    document = docx.Document()
    for line in text.split("\n"):
        document.add_paragraph(line)
    path = tmp_path / "same.docx"
    document.save(path)

    draft_store.clear()
    parsed = client.post(
        "/api/parse",
        files={"file": ("same.docx", path.read_bytes(),
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    ).json()
    classified = client.post("/api/classify", json=parsed).json()
    docx_xml = client.post(
        "/api/generate", json={"questions": classified["questions"]}
    ).json()["xml"]

    quick_xml = convert(client, text).json()["xml"]
    draft_store.clear()

    assert quick_xml == docx_xml


# -- the document flow is untouched ---------------------------------------


def test_quick_convert_leaves_a_document_review_alone(client, scripted_ai, sample_docx):
    """A Quick Convert mid-review must not disturb the uploaded draft."""
    draft_store.clear()
    parsed = client.post(
        "/api/parse",
        files={"file": ("sample.docx", sample_docx.read_bytes(),
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    ).json()
    client.post("/api/classify", json=parsed)
    before = client.get("/api/questions").json()

    convert(client, MULTI)

    after = client.get("/api/questions").json()
    assert after == before
    draft_store.clear()


def test_quick_convert_does_not_record_corrections(client, scripted_ai):
    """Corrections belong to the document under review, not to a paste."""
    from app.classify.corrections import correction_memory

    correction_memory.clear()
    convert(client, MULTI)
    assert correction_memory.recent() == []


def test_both_pages_are_served(client):
    assert client.get("/").status_code == 200
    assert "Decipher Survey Base Generator" in client.get("/").text

    quick = client.get("/quick")
    assert quick.status_code == 200
    assert "Quick Convert" in quick.text


def test_the_pages_link_to_each_other(client):
    assert 'href="/quick"' in client.get("/").text
    assert 'href="/"' in client.get("/quick").text
