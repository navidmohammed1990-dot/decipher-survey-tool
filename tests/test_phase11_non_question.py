"""Phase 11 — programmer content is not a question.

The failing input: a derived-variable definition forced into a radio, where
one row survived and that row was the variable's own name.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.classify.classifier import SYSTEM_PROMPT, interpret_response
from app.classify.lines import question_lines
from app.classify.ollama import OllamaClient
from app.classify.paste import join_cells, split_questions
from app.classify.seed_library import SEED_EXAMPLES, prompt_prefix
from app.generate.xml_generator import generate_question, generate_questions
from app.main import app
from app.models.survey import NON_QUESTION_ELEMENTS, SUPPORTED_ELEMENTS, Question

AGE_BANDS = """[Please create the following variable for datafile and auto code based on S2 Age]:
S2_AGE BANDS
Under 18 years  1
18 to 24 years  2
25 to 29 years  3
30 to 39 years  4
40 to 49 years  5
50 to 59 years  6
60 to 69 years  7
70 years or more   8"""

#: Differently worded, same shape — proves this generalises past one example.
POSTCODE_ROLLUP = """(Programmer: derive the following from Q4 postcode and code automatically)
Q4_REGION_GROUP
Metropolitan  1
Regional  2
Remote  3
Not stated  9"""

#: A real eight-option question. Must never be caught by the above.
REAL_EIGHT_OPTIONS = """Q7. Which of the following age groups do you belong to?
Please select one.
Under 18 years  1
18 to 24 years  2
25 to 29 years  3
30 to 39 years  4
40 to 49 years  5
50 to 59 years  6
60 to 69 years  7
70 years or more   8"""


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def lines_of(text):
    blocks, _ = split_questions(text)
    return blocks[0].lines


def classify(payload, text, label="Q1", threshold=0.75):
    return interpret_response(payload, label, lines_of(text), threshold)


NON_QUESTION = {"element": "not_a_question", "confidence": 0.93,
                "notes": "derived variable definition addressed to the programmer"}


# -- Part A: the outcome exists and generates nothing ----------------------


def test_not_a_question_is_a_sibling_element():
    assert "not_a_question" in SUPPORTED_ELEMENTS
    assert NON_QUESTION_ELEMENTS == {"not_a_question"}


def test_the_age_banding_example_produces_no_xml():
    """Checklist: no malformed radio element."""
    question = classify(NON_QUESTION, AGE_BANDS).question

    assert question.element == "not_a_question"
    assert generate_question(question) == ""
    assert generate_questions([question]) == ""


def test_nothing_is_put_through_question_shaped_processing():
    question = classify(NON_QUESTION, AGE_BANDS).question

    assert question.options == []
    assert question.rows == []
    assert question.cols == []
    assert question.title == []


def test_the_original_text_is_kept_for_reference():
    """The programmer still needs to read and copy what they pasted."""
    question = classify(NON_QUESTION, AGE_BANDS).question

    assert len(question.routing_notes) == 10
    assert question.routing_notes[1] == "S2_AGE BANDS"
    assert question.routing_notes[-1].startswith("70 years or more")


def test_a_non_question_is_not_flagged_for_review():
    """It is a definite answer, not an uncertain one."""
    assert classify(NON_QUESTION, AGE_BANDS).question.needs_review is False


def test_a_differently_worded_instruction_is_also_handled():
    """Checklist: generalises rather than matching one example verbatim."""
    question = classify(NON_QUESTION, POSTCODE_ROLLUP).question

    assert generate_questions([question]) == ""
    assert any("Q4_REGION_GROUP" in note for note in question.routing_notes)


def prompt_text():
    """The prompt with wrapping removed, so tests match wording not layout."""
    return " ".join(SYSTEM_PROMPT.split())


def test_the_prompt_offers_cues_as_evidence_not_rules():
    text = prompt_text()

    assert "not_a_question" in text
    assert "Evidence, not rules" in text
    for cue in ("auto code based on", "S2_AGE BANDS", "already-asked question"):
        assert cue in text


def test_a_block_that_asks_something_stays_a_question():
    """The prompt must not let a stray programmer note void a real question."""
    assert "put programmer notes in routing_lines instead" in prompt_text()


def test_non_question_blocks_are_skipped_in_a_mixed_batch():
    real = Question(label="Q1", element="radio",
                    title=[], options=[], needs_review=False)
    meta = classify(NON_QUESTION, AGE_BANDS, label="Q2").question

    xml = generate_questions([
        classify({"element": "radio", "title_lines": [0], "option_lines": [2, 3, 4, 5, 6, 7, 8, 9],
                  "comment_lines": [1], "confidence": 0.9, "notes": "ages"},
                 REAL_EIGHT_OPTIONS).question,
        meta,
    ])
    assert xml.count("<suspend/>") == 1, "only the real question is rendered"
    assert "S2_AGE BANDS" not in xml


# -- the false-positive guard ---------------------------------------------


def test_a_real_eight_option_question_still_works():
    """Checklist: a genuine long option list must not be caught by any of this."""
    outcome = classify(
        {"element": "radio", "title_lines": [0], "comment_lines": [1],
         "option_lines": [2, 3, 4, 5, 6, 7, 8, 9], "confidence": 0.94, "notes": "age bands"},
        REAL_EIGHT_OPTIONS,
    )
    question = outcome.question

    assert question.element == "radio"
    assert len(question.options) == 8
    assert question.needs_review is False, "a complete option list raises no warning"
    assert outcome.warnings == []

    xml = generate_question(question)
    assert xml.count("<row ") == 8
    assert '<row label="r1" value="1">Under 18 years</row>' in xml
    assert '<row label="r8" value="8">70 years or more</row>' in xml


def test_the_two_shapes_are_distinguishable_by_their_lines():
    """The only structural difference is the wording, which is the point."""
    banding = [line.text for line in lines_of(AGE_BANDS)]
    real = [line.text for line in lines_of(REAL_EIGHT_OPTIONS)]

    assert banding[2:] == real[2:], "identical band lines in both"
    assert banding[0].startswith("[Please create")
    assert real[0].startswith("Which of the following age groups")


# -- Part B: the dropped-option guard --------------------------------------


def test_one_row_from_eight_candidates_is_flagged():
    """Checklist: an implausible option count warns instead of exporting."""
    outcome = classify(
        {"element": "radio", "title_lines": [0], "option_lines": [1],
         "confidence": 0.8, "notes": "single select"},
        AGE_BANDS,
    )

    assert outcome.question.needs_review is True
    assert "look like options and were left out" in outcome.question.ai_notes
    assert "1 option(s) were taken" in outcome.question.ai_notes


def test_the_guard_does_not_fire_on_a_complete_list():
    outcome = classify(
        {"element": "radio", "title_lines": [0], "comment_lines": [1],
         "option_lines": [2, 3, 4, 5, 6, 7, 8, 9], "confidence": 0.9, "notes": "ok"},
        REAL_EIGHT_OPTIONS,
    )
    assert not any("left out" in w for w in outcome.warnings)


def test_the_guard_ignores_a_couple_of_stray_lines():
    """Two unclaimed lines is normal; it takes a real gap to warrant a flag."""
    text = "Q1. Pick one\nPlease select one.\nYes\nNo\nSome trailing note"
    outcome = classify(
        {"element": "radio", "title_lines": [0], "option_lines": [2, 3],
         "confidence": 0.9, "notes": "ok"},
        text,
    )
    assert not any("left out" in w for w in outcome.warnings)


def test_the_guard_ignores_long_prose_lines():
    """Unclaimed paragraphs are not missing options."""
    prose = " ".join(["a"] * 60)
    text = f"Q1. Pick one\nYes\nNo\n{prose}\n{prose}\n{prose}"
    outcome = classify(
        {"element": "radio", "title_lines": [0], "option_lines": [1, 2],
         "confidence": 0.9, "notes": "ok"},
        text,
    )
    assert not any("left out" in w for w in outcome.warnings)


def test_the_guard_ignores_unclaimed_routing_lines():
    text = ("Q1. Pick one\nYes\nNo\nTERMINATE IF Q1 = 2\n"
            "ASK ALL, SC\nRANDOMIZE OPTIONS")
    outcome = classify(
        {"element": "radio", "title_lines": [0], "option_lines": [1, 2],
         "confidence": 0.9, "notes": "ok"},
        text,
    )
    assert not any("left out" in w for w in outcome.warnings)


def test_the_guard_does_not_apply_to_open_ended_elements():
    text = "Q1. Tell us why\nline one\nline two\nline three\nline four"
    outcome = classify(
        {"element": "textarea", "title_lines": [0], "confidence": 0.9, "notes": "open"},
        text,
    )
    assert not any("left out" in w for w in outcome.warnings)


# -- the space-aligned code bug found while investigating -------------------


@pytest.mark.parametrize(
    "line,expected",
    [
        ("Under 18 years  1", "Under 18 years | 1"),
        ("70 years or more   8", "70 years or more | 8"),
        ("Metropolitan  1", "Metropolitan | 1"),
    ],
)
def test_column_aligned_codes_are_recovered(line, expected):
    """Word pastes table columns as runs of spaces as often as tabs."""
    assert join_cells(line) == expected


@pytest.mark.parametrize(
    "line",
    ["I have lived here 20 years", "Yes 1", "Please select one.",
     "Some sentence.  Another sentence.", "Brand A"],
)
def test_ordinary_text_keeps_its_numbers(line):
    """A single space, or a number mid-sentence, is not a column."""
    assert join_cells(line) == line.strip()


def test_the_recovered_codes_reach_the_generated_xml():
    outcome = classify(
        {"element": "radio", "title_lines": [0], "comment_lines": [1],
         "option_lines": [2, 3, 4, 5, 6, 7, 8, 9], "confidence": 0.9, "notes": "ages"},
        REAL_EIGHT_OPTIONS,
    )
    assert [o.code for o in outcome.question.options] == list("12345678")


# -- Part C: the seed library ---------------------------------------------


def test_the_age_banding_example_is_seeded():
    assert any(e.correct.get("element") == "not_a_question" for e in SEED_EXAMPLES)

    prefix = prompt_prefix()
    assert "S2_AGE BANDS" in prefix
    assert "not_a_question" in prefix


def test_the_seed_library_stays_small():
    """Every entry is paid for on every call, in prompt-eval seconds."""
    assert len(prompt_prefix()) < 1200


def test_seeded_examples_travel_with_every_call(monkeypatch, client):
    """Unlike document corrections, these are not session-scoped."""
    from app.api import routes_classify
    from app.classify.corrections import correction_memory

    seen = []

    class Capturing(OllamaClient):
        def generate_json(self, system, prompt):
            seen.append(system)
            return {"element": "radio", "title_lines": [0], "confidence": 0.9, "notes": "ok"}

        def status(self):
            return {"available": True, "reachable": True, "model_installed": True,
                    "url": "s", "model": "s", "installed_models": ["s"], "detail": "Ready."}

    correction_memory.clear()
    monkeypatch.setattr(routes_classify, "build_client", Capturing)
    client.post("/api/quick-convert", json={"text": "Q1. Pick one\nYes\nNo"})

    assert "S2_AGE BANDS" in seen[0], "the seed example is present with no corrections at all"


# -- end to end over HTTP --------------------------------------------------


@pytest.fixture
def stub_ai(monkeypatch):
    from app.api import routes_classify

    def install(payload):
        class Stub(OllamaClient):
            def generate_json(self, system, prompt):
                return payload

            def status(self):
                return {"available": True, "reachable": True, "model_installed": True,
                        "url": "s", "model": "s", "installed_models": ["s"], "detail": "Ready."}

        monkeypatch.setattr(routes_classify, "build_client", Stub)

    return install


def test_quick_convert_reports_a_programmer_instruction(client, stub_ai):
    stub_ai(NON_QUESTION)
    body = client.post("/api/quick-convert", json={"text": AGE_BANDS}).json()

    assert body["xml"] == "", "no XML at all"
    assert body["well_formed"] is True
    assert any("programmer instruction" in w for w in body["warnings"])
    assert body["questions"][0]["element"] == "not_a_question"
    assert len(body["questions"][0]["routing_notes"]) == 10


def test_quick_convert_still_builds_a_real_question(client, stub_ai):
    stub_ai({"element": "radio", "title_lines": [0], "comment_lines": [1],
             "option_lines": [2, 3, 4, 5, 6, 7, 8, 9], "confidence": 0.94, "notes": "ages"})
    body = client.post("/api/quick-convert", json={"text": REAL_EIGHT_OPTIONS}).json()

    assert body["xml"].count("<row ") == 8
    assert '<row label="r8" value="8">70 years or more</row>' in body["xml"]
    assert body["warnings"] == []
