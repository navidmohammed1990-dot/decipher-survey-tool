"""Phase 7 — AI-led line classification.

The architectural claim under test: hints feed the model's judgment and never
bypass it. The load-bearing case is a routing line with no colour and no
keyword match — if that works, the pipeline is not living off pattern coverage.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.classify.classifier import SYSTEM_PROMPT, build_prompt, interpret_response
from app.classify.corrections import Correction, correction_memory
from app.classify.lines import question_lines
from app.classify.ollama import OllamaClient
from app.generate.xml_generator import generate_question
from app.main import app
from app.parsing.docx_parser import parse_docx
from app.store import draft_store
from tests.fixtures.build_tasmania import (
    build_format_variations,
    build_house_convention,
    build_tasmania,
)


@pytest.fixture(scope="module")
def house(tmp_path_factory):
    return parse_docx(build_house_convention(tmp_path_factory.mktemp("h") / "house.docx"))


@pytest.fixture(scope="module")
def variations(tmp_path_factory):
    return parse_docx(build_format_variations(tmp_path_factory.mktemp("v") / "var.docx"))


@pytest.fixture(scope="module")
def tasmania(tmp_path_factory):
    return parse_docx(build_tasmania(tmp_path_factory.mktemp("t") / "tas.docx"))


def lines_for(parsed, label):
    return question_lines(parsed, next(q for q in parsed.questions if q.label == label))


def classify(payload, parsed, label, threshold=0.75):
    lines = lines_for(parsed, label)
    return interpret_response(payload, label, lines, threshold)


# -- the prompt puts judgment first ---------------------------------------


def test_the_prompt_frames_hints_as_evidence_not_rules():
    assert "Hints are not rules" in SYSTEM_PROMPT
    assert "make the" in SYSTEM_PROMPT and "judgment call yourself" in SYSTEM_PROMPT
    for role in ("title", "comment", "option", "routing", "type_signal"):
        assert role in SYSTEM_PROMPT


def test_lines_reach_the_model_with_hints_attached(tasmania):
    prompt = build_prompt("Q1.2", lines_for(tasmania, "Q1.2"))

    assert '"Male | 1"  [hints: is_table_row=true, trailing_numeric_code=1]' in prompt
    assert "color_hint=red" in prompt
    assert "[hints: none]" in prompt, "a line with nothing observed still gets offered"


# -- the load-bearing case: no colour, no keyword --------------------------


def test_a_house_convention_routing_line_carries_no_hints(house):
    """Precondition: nothing about this line is pattern-detectable.

    If it were, the next test would prove nothing about generalisation.
    """
    line = lines_for(house, "Q1")[5]

    assert line.text.startswith("Respondents choosing Dissatisfied")
    assert line.features.matches_routing_keyword is False
    assert line.features.is_colored is False
    assert line.features.matches_type_tag_pattern is False
    assert line.features.as_prompt_hints() == "none"


def test_judgment_alone_can_place_a_hintless_routing_line(house):
    """The model's call is honoured with no corroborating pattern at all."""
    outcome = classify(
        {
            "element": "radio",
            "title_lines": [0], "comment_lines": [1], "option_lines": [2, 3, 4],
            "routing_lines": [5],
            "confidence": 0.88,
            "notes": "Line 5 addresses the scripter about which module to show, not the respondent.",
        },
        house, "Q1",
    )
    question = outcome.question

    assert question.routing_notes == [
        "Respondents choosing Dissatisfied should be shown the follow-up module "
        "before continuing to section 2."
    ]
    assert [o.raw_text for o in question.options] == [
        "Very satisfied", "Satisfied", "Dissatisfied",
    ]
    assert question.needs_review is False, "judgment without pattern backing is not penalised"


def test_a_hintless_routing_line_never_reaches_the_xml(house):
    outcome = classify(
        {
            "element": "radio", "title_lines": [0], "comment_lines": [1],
            "option_lines": [2, 3, 4], "routing_lines": [5], "confidence": 0.88,
            "notes": "line 5 is a scripter instruction",
        },
        house, "Q1",
    )
    xml_text = generate_question(outcome.question)

    assert "follow-up module" not in xml_text
    assert xml_text.count("<row") == 3


def test_an_unexplained_hintless_routing_call_is_flagged(house):
    """Judgment is trusted when reasoned. Silence is not reasoning."""
    outcome = classify(
        {
            "element": "radio", "title_lines": [0], "comment_lines": [1],
            "option_lines": [2, 3, 4], "routing_lines": [5], "confidence": 0.95,
            "notes": "",
        },
        house, "Q1",
    )
    assert outcome.question.needs_review is True
    assert "no routing hint" in outcome.question.ai_notes


# -- planted disagreement --------------------------------------------------


def test_a_real_option_that_reads_like_an_instruction_is_flagged(house):
    """"Randomly assigned to me by my employer" is a genuine option that trips
    the keyword hint. Neither side is silently believed."""
    line = lines_for(house, "Q2")[3]
    assert line.features.matches_routing_keyword is True

    outcome = classify(
        {
            "element": "radio", "title_lines": [0], "option_lines": [1, 2, 3],
            "confidence": 0.93, "notes": "three ways a provider might be chosen",
        },
        house, "Q2",
    )

    assert outcome.question.needs_review is True
    assert "looks like routing text by keyword" in outcome.question.ai_notes
    assert "please check" in outcome.question.ai_notes
    # Not auto-corrected: the option is still there for the programmer to judge.
    assert len(outcome.question.options) == 3


def test_a_type_tag_contradicting_the_element_is_flagged(tasmania):
    """Q1.2 is marked SC but the model called it a checkbox."""
    outcome = classify(
        {
            "element": "checkbox", "title_lines": [1], "option_lines": [2, 3, 4],
            "routing_lines": [0, 5], "confidence": 0.9, "notes": "looks multi-select",
        },
        tasmania, "Q1.2",
    )

    assert outcome.question.needs_review is True
    assert "marks this question as 'SC'" in outcome.question.ai_notes


def test_a_matching_type_tag_raises_no_complaint(tasmania):
    outcome = classify(
        {
            "element": "radio", "title_lines": [1], "option_lines": [2, 3, 4],
            "routing_lines": [0, 5], "confidence": 0.95, "notes": "single select per SC tag",
        },
        tasmania, "Q1.2",
    )
    assert outcome.question.needs_review is False


def test_a_grid_may_override_the_type_tag(sample_docx):
    """The prompt allows a grid to outrank SC/MC, so this is not a conflict."""
    parsed = parse_docx(sample_docx)
    outcome = classify(
        {
            "element": "radio_grid", "title_lines": [0], "comment_lines": [1],
            "col_lines": [3, 4], "row_lines": [5, 6], "subject_type": "statement",
            "confidence": 0.9, "notes": "row statements by scale columns",
        },
        parsed, "Q6",
    )
    assert outcome.question.needs_review is False


def test_a_type_tag_is_ignored_when_there_are_no_options(tasmania):
    """Q1.1 is a numeric entry; its NUM tag must not fight the element."""
    outcome = classify(
        {
            "element": "number", "title_lines": [1], "routing_lines": [0],
            "confidence": 0.9, "notes": "age, free numeric entry",
        },
        tasmania, "Q1.1",
    )
    assert outcome.question.needs_review is False
    assert "marks this question" not in outcome.question.ai_notes


# -- format independence ---------------------------------------------------


@pytest.mark.parametrize(
    "label,option_indices",
    [("Q1", [1, 2, 3, 4]), ("Q2", [1, 2, 3, 4]), ("Q3", [1, 2, 3, 4])],
)
def test_three_formats_produce_equivalent_output(variations, label, option_indices):
    """Table with codes, numbered list, and bulleted list with parenthetical
    codes must all land on the same rows."""
    outcome = classify(
        {
            "element": "checkbox", "title_lines": [0], "option_lines": option_indices,
            "confidence": 0.9, "notes": "select all",
        },
        variations, label,
    )
    xml_text = generate_question(outcome.question)

    assert '<row label="r1">Brand A</row>' in xml_text
    assert '<row label="r2">Brand B</row>' in xml_text
    assert '<row label="r97" open="1" openSize="25" randomize="0">Other, please specify</row>' in xml_text
    assert '<row label="r99" randomize="0" exclusive="1">None of these</row>' in xml_text


def test_the_three_formats_agree_exactly(variations):
    payload = {"element": "checkbox", "title_lines": [0], "option_lines": [1, 2, 3, 4],
               "confidence": 0.9, "notes": "select all"}
    outputs = set()
    for label in ("Q1", "Q2", "Q3"):
        question = classify(payload, variations, label).question
        question.label = "QX"
        outputs.add(generate_question(question))

    assert len(outputs) == 1, "the same question written three ways must generate identically"


# -- correction learning ---------------------------------------------------


class RecordingClient(OllamaClient):
    """Captures the system prompt each call receives."""

    def __init__(self, payload):
        super().__init__()
        self.payload = payload
        self.systems: list[str] = []

    def generate_json(self, system, prompt):
        self.systems.append(system)
        return self.payload

    def is_available(self):
        return True


@pytest.fixture(autouse=True)
def _clean_memory():
    correction_memory.clear()
    draft_store.clear()
    yield
    correction_memory.clear()
    draft_store.clear()


def test_a_correction_is_carried_into_later_calls(house):
    from app.classify.classifier import classify_question

    correction_memory.use_document("house.docx")
    correction_memory.record(Correction(
        label="Q1",
        original_lines=["How satisfied are you?", "Respondents choosing Dissatisfied should..."],
        ai_said={"element": "radio", "option_lines": [2, 3, 4, 5]},
        sp_corrected_to={"element": "radio", "option_lines": [2, 3, 4],
                         "routing_lines": [5]},
    ))

    client = RecordingClient({"element": "radio", "title_lines": [0],
                              "option_lines": [1, 2], "routing_lines": [3],
                              "confidence": 0.9, "notes": "learned from the correction"})
    classify_question("Q2", lines_for(house, "Q2"), client)

    system = client.systems[0]
    assert "The survey programmer corrected an earlier question" in system
    assert "Learn from this pattern for the rest of this document" in system
    assert system.endswith(SYSTEM_PROMPT), "corrections prepend, they do not replace"


def test_corrections_do_not_leak_between_documents():
    correction_memory.use_document("client_a.docx")
    correction_memory.record(Correction(
        label="Q1", original_lines=["x"],
        ai_said={"element": "radio"}, sp_corrected_to={"element": "checkbox"},
    ))
    assert correction_memory.prompt_prefix() != ""

    correction_memory.use_document("client_b.docx")
    assert correction_memory.prompt_prefix() == ""


def test_only_real_changes_are_remembered():
    correction_memory.use_document("doc.docx")
    unchanged = Correction(label="Q1", original_lines=["x"],
                           ai_said={"element": "radio"}, sp_corrected_to={"element": "radio"})
    assert correction_memory.record(unchanged) is False
    assert correction_memory.prompt_prefix() == ""


def test_only_the_most_recent_corrections_are_kept():
    """Phase 17 widened the pool that is held; what is *sent* is still three."""
    from app.classify.corrections import CORRECTION_POOL, MAX_CORRECTIONS

    correction_memory.use_document("doc.docx")
    for number in range(CORRECTION_POOL + 6):
        correction_memory.record(Correction(
            label=f"Q{number}", original_lines=["x"],
            ai_said={"element": "radio"}, sp_corrected_to={"element": "checkbox"},
        ))

    kept = correction_memory.recent()
    assert len(kept) == CORRECTION_POOL
    assert kept[-1].label == f"Q{CORRECTION_POOL + 5}"
    assert correction_memory.prompt_prefix().count("Example") <= MAX_CORRECTIONS


def test_re_editing_a_question_supersedes_its_earlier_correction():
    correction_memory.use_document("doc.docx")
    for element in ("checkbox", "select"):
        correction_memory.record(Correction(
            label="Q1", original_lines=["x"],
            ai_said={"element": "radio"}, sp_corrected_to={"element": element},
        ))

    kept = correction_memory.recent()
    assert len(kept) == 1
    assert kept[0].sp_corrected_to["element"] == "select"


# -- capture over HTTP -----------------------------------------------------


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_an_sp_edit_is_captured_as_a_correction(client, house):
    from app.models.survey import ClassificationTrace, Question, QuestionDraft

    correction_memory.use_document("house.docx")
    draft_store.replace(QuestionDraft(questions=[Question(
        label="Q1", element="radio",
        trace=ClassificationTrace(
            lines=["How satisfied are you?", "Respondents choosing Dissatisfied..."],
            ai_payload={"element": "radio", "option_lines": [2, 3, 4, 5]},
        ),
    )]))

    assert client.patch("/api/questions/Q1", json={"element": "checkbox"}).status_code == 200

    body = client.get("/api/corrections").json()
    assert body["count"] == 1
    assert body["corrections"][0]["sp_corrected_to"]["element"] == "checkbox"
