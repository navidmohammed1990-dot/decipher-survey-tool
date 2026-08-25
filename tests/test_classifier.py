"""Phase 2 — AI classification, its index mapping, and its fallback."""

from __future__ import annotations

import pytest

from app.classify.classifier import (
    FALLBACK_NOTE,
    SYSTEM_PROMPT,
    build_prompt,
    classify_document,
    classify_question,
    fallback_question,
    interpret_response,
)
from app.classify.lines import question_lines
from app.classify.ollama import OllamaClient, OllamaError
from app.models.survey import SUPPORTED_ELEMENTS


class FakeClient(OllamaClient):
    """Returns canned JSON, or raises, without touching the network."""

    def __init__(self, payload=None, error=None):
        super().__init__()
        self.payload = payload
        self.error = error
        self.calls = []

    def generate_json(self, system, prompt):
        self.calls.append((system, prompt))
        if self.error:
            raise self.error
        return self.payload


@pytest.fixture
def q5_lines(parsed_sample):
    q5 = next(q for q in parsed_sample.questions if q.label == "Q5")
    return question_lines(parsed_sample, q5)


@pytest.fixture
def q6_lines(parsed_sample):
    q6 = next(q for q in parsed_sample.questions if q.label == "Q6")
    return question_lines(parsed_sample, q6)


# -- line construction ----------------------------------------------------


def test_lines_strip_the_question_label_from_the_title(q5_lines):
    assert q5_lines[0].text.startswith("Which of the following brands")
    assert "Q5" not in q5_lines[0].text


def test_lines_strip_typed_list_markers(parsed_sample):
    q7 = next(q for q in parsed_sample.questions if q.label == "Q7")
    lines = question_lines(parsed_sample, q7)

    assert [line.text for line in lines[1:4]] == ["Price", "Quality", "Availability"]
    assert [line.marker for line in lines[1:4]] == ["1.", "2.", "3."]


def test_lines_keep_phase_1_formatting(q5_lines):
    title = q5_lines[0]
    assert [r.text for r in title.runs if r.bold] == ["purchased"]
    assert [r.text for r in title.runs if r.italic] == ["6 months"]


def test_grid_tables_become_row_and_column_lines(q6_lines):
    kinds = {line.kind for line in q6_lines}
    assert "table_col" in kinds and "table_row" in kinds

    cols = [line.text for line in q6_lines if line.kind == "table_col"]
    rows = [line.text for line in q6_lines if line.kind == "table_row"]
    # The stub header ("Statement") is offered as a candidate too; deciding it
    # is not a scale point is the classifier's judgment, not the parser's.
    assert cols == ["Statement", "Agree", "Disagree"]
    assert rows == ["The brand is good value", "The brand is easy to find"]


def test_prompt_numbers_every_line(q5_lines):
    prompt = build_prompt("Q5", q5_lines)
    assert "Question label: Q5" in prompt
    for line in q5_lines:
        assert f"{line.index}: " in prompt


def test_system_prompt_lists_every_supported_element():
    for element in SUPPORTED_ELEMENTS:
        assert element in SYSTEM_PROMPT


# -- index mapping --------------------------------------------------------


def test_indices_map_back_onto_parsed_text_and_formatting(q5_lines):
    outcome = interpret_response(
        {
            "element": "checkbox",
            "title_lines": [0],
            "comment_lines": [1],
            "option_lines": [2, 3, 4, 5],
            "confidence": 0.94,
            "notes": "select all that apply",
        },
        "Q5",
        q5_lines,
        0.75,
    )
    question = outcome.question

    assert question.element == "checkbox"
    assert question.title_text().startswith("Which of the following brands")
    assert question.comment_text() == "Please select all that apply."
    assert [o.raw_text for o in question.options] == [
        "Brand A", "Brand B", "Brand C", "None of these",
    ]
    # Formatting came from Phase 1, not from the model's JSON.
    assert [r.text for r in question.title if r.bold] == ["purchased"]
    assert question.confidence == 0.94
    assert question.needs_review is False


def test_grid_rows_and_cols_are_mapped(q6_lines):
    outcome = interpret_response(
        {
            "element": "radio_grid",
            "title_lines": [0],
            "comment_lines": [1],
            "row_lines": [5, 6],
            "col_lines": [3, 4],
            "confidence": 0.9,
        },
        "Q6",
        q6_lines,
        0.75,
    )
    question = outcome.question

    assert [r.raw_text for r in question.rows] == [
        "The brand is good value", "The brand is easy to find",
    ]
    assert [c.raw_text for c in question.cols] == ["Agree", "Disagree"]


def test_multiple_title_lines_are_joined(q5_lines):
    outcome = interpret_response(
        {"element": "radio", "title_lines": [0, 1], "option_lines": [2], "confidence": 0.9},
        "Q5", q5_lines, 0.75,
    )
    assert "Please select all that apply." in outcome.question.title_text()


def test_out_of_range_indices_are_discarded_not_crashed(q5_lines):
    outcome = interpret_response(
        {"element": "radio", "title_lines": [0], "option_lines": [2, 99, -1], "confidence": 0.9},
        "Q5", q5_lines, 0.75,
    )
    assert [o.raw_text for o in outcome.question.options] == ["Brand A"]
    assert any("unknown lines" in w for w in outcome.warnings)
    assert outcome.question.needs_review is True


def test_duplicate_indices_are_collapsed(q5_lines):
    outcome = interpret_response(
        {"element": "radio", "title_lines": [0], "option_lines": [2, 2, 3], "confidence": 0.9},
        "Q5", q5_lines, 0.75,
    )
    assert [o.raw_text for o in outcome.question.options] == ["Brand A", "Brand B"]


def test_missing_title_defaults_to_the_first_line(q5_lines):
    outcome = interpret_response(
        {"element": "radio", "option_lines": [2], "confidence": 0.9}, "Q5", q5_lines, 0.75
    )
    assert outcome.question.title_text().startswith("Which of the following")
    assert outcome.question.needs_review is True


@pytest.mark.parametrize("element", ["", "grid", "dropdown", None, 42])
def test_unsupported_element_is_rejected(q5_lines, element):
    with pytest.raises(ValueError):
        interpret_response({"element": element}, "Q5", q5_lines, 0.75)


def test_confidence_is_clamped_and_defaulted(q5_lines):
    for raw, expected in [(1.5, 1.0), (-2, 0.0), ("nonsense", 0.0), (None, 0.0)]:
        outcome = interpret_response(
            {"element": "radio", "title_lines": [0], "option_lines": [2], "confidence": raw},
            "Q5", q5_lines, 0.75,
        )
        assert outcome.question.confidence == expected


def test_low_confidence_flags_for_review(q5_lines):
    outcome = interpret_response(
        {"element": "radio", "title_lines": [0], "option_lines": [2], "confidence": 0.5},
        "Q5", q5_lines, 0.75,
    )
    assert outcome.question.needs_review is True


def test_threshold_is_respected(q5_lines):
    payload = {"element": "radio", "title_lines": [0], "option_lines": [2], "confidence": 0.5}
    assert interpret_response(payload, "Q5", q5_lines, 0.4).question.needs_review is False
    assert interpret_response(payload, "Q5", q5_lines, 0.9).question.needs_review is True


# -- structural guards ----------------------------------------------------


def test_option_element_without_options_is_flagged(q5_lines):
    outcome = interpret_response(
        {"element": "checkbox", "title_lines": [0], "option_lines": [], "confidence": 0.99},
        "Q5", q5_lines, 0.75,
    )
    assert outcome.question.needs_review is True
    assert "has no options" in outcome.question.ai_notes


def test_grid_without_rows_or_columns_is_flagged(q6_lines):
    outcome = interpret_response(
        {"element": "radio_grid", "title_lines": [0], "confidence": 0.99}, "Q6", q6_lines, 0.75
    )
    assert outcome.question.needs_review is True
    assert "no rows" in outcome.question.ai_notes
    assert "no columns" in outcome.question.ai_notes


# -- fallback -------------------------------------------------------------


def test_fallback_is_title_first_line_rest_options(q5_lines):
    question = fallback_question("Q5", q5_lines)

    assert question.element == "radio"
    assert question.title_text().startswith("Which of the following")
    assert len(question.options) == len(q5_lines) - 1
    assert question.confidence == 0.0
    assert question.needs_review is True
    assert question.ai_notes == FALLBACK_NOTE


@pytest.mark.parametrize(
    "error",
    [OllamaError("connection refused"), OllamaError("not valid JSON")],
)
def test_unreachable_model_degrades_rather_than_crashing(q5_lines, error):
    outcome = classify_question("Q5", q5_lines, FakeClient(error=error))

    assert outcome.used_fallback is True
    assert outcome.question.needs_review is True
    assert outcome.question.confidence == 0.0


def test_unusable_json_degrades_to_fallback(q5_lines):
    outcome = classify_question("Q5", q5_lines, FakeClient(payload={"element": "nonsense"}))

    assert outcome.used_fallback is True
    assert outcome.question.ai_notes == FALLBACK_NOTE


def test_a_question_with_no_content_does_not_crash():
    outcome = classify_question("Q9", [], FakeClient(payload={}))
    assert outcome.used_fallback is True
    assert outcome.question.label == "Q9"


# -- whole document -------------------------------------------------------


def test_classify_document_calls_the_model_once_per_question(parsed_sample):
    client = FakeClient(payload={
        "element": "radio", "title_lines": [0], "option_lines": [1], "confidence": 0.9,
    })
    outcomes = classify_document(parsed_sample, client)

    assert len(outcomes) == 4, "one per question, preamble excluded"
    assert len(client.calls) == 4


def test_classify_document_skips_the_preamble(parsed_sample):
    client = FakeClient(payload={"element": "radio", "title_lines": [0], "confidence": 0.9})
    labels = [o.question.label for o in classify_document(parsed_sample, client)]

    assert labels == ["S1", "Q5", "Q6", "Q7"]


def test_real_unreachable_ollama_falls_back(parsed_sample):
    """No mocking: point at a dead port and confirm it degrades, not crashes."""
    client = OllamaClient(base_url="http://127.0.0.1:1", timeout=2.0)
    outcomes = classify_document(parsed_sample, client)

    assert len(outcomes) == 4
    assert all(o.used_fallback for o in outcomes)
    assert all(o.question.needs_review for o in outcomes)
    assert all(o.question.ai_notes == FALLBACK_NOTE for o in outcomes)
