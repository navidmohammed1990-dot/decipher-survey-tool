"""Phase 17 — both knowledge sources feed live classification, bounded.

Before this, the reference dataset was regression-only: the model never saw
any of it. The correction library did reach live calls, but sent its three
most *recent* entries regardless of what was being classified.

Now both are selected by relevance to the question in front of the model, and
both are capped, so the prompt is the same size whether the sources hold ten
examples or ten thousand.
"""

from __future__ import annotations

import pytest

from app.classify import reference
from app.classify.corrections import (
    CORRECTION_POOL,
    MAX_CORRECTIONS,
    Correction,
    CorrectionMemory,
)
from app.classify.relevance import most_relevant, score

GENDER = [
    "Which of the following best describes your gender identity?",
    "Male | 1",
    "Female | 2",
    "Other | 97",
]
OPEN_END = ["What is the main reason you feel this way about Australia Post?"]
GRID = [
    "For each statement below, tell us how strongly you agree or disagree",
    "Strongly Disagree | 1",
    "Agree | 4",
    "Delivers parcels quickly",
    "Delivers parcels on time",
]


# -- the relevance signal --------------------------------------------------


def test_a_similar_question_scores_above_an_unrelated_one():
    similar = ["What gender do you identify as?", "Male | 1", "Female | 2"]
    assert score(GENDER, similar) > score(GENDER, GRID)
    assert score(GENDER, GRID) > score(GENDER, OPEN_END)


def test_an_unrelated_example_is_dropped_rather_than_ranked_last():
    """Below the floor, no example is better than a misleading one."""
    chosen = most_relevant(OPEN_END, [(GRID, "grid")], limit=2)
    assert chosen == []


def test_selection_returns_the_best_first():
    close = ["What gender do you identify as?", "Male | 1", "Female | 2"]
    chosen = most_relevant(
        GENDER, [(GRID, "grid"), (close, "close")], limit=2
    )
    assert chosen[0] == "close"


# -- the dataset as a live source -----------------------------------------


def test_the_dataset_is_available_as_prompt_examples():
    assert len(reference.examples()) >= 20


def test_a_dataset_matching_input_gets_its_own_precedent():
    """The checklist's case: something like `single_select_coded_table`."""
    prefix = reference.prompt_prefix(GENDER)

    assert "single_select_coded_table" in prefix
    assert '"element": "radio"' in prefix
    assert "precedent, not a rule" in prefix


def test_an_open_text_question_does_not_get_a_grid_example():
    prefix = reference.prompt_prefix(OPEN_END)

    assert "two_table_grid" not in prefix
    assert "checkbox_grid_basic" not in prefix


def test_no_more_than_the_cap_is_ever_sent():
    prefix = reference.prompt_prefix(GENDER)
    assert prefix.count("Correct answer:") <= reference.MAX_REFERENCE_EXAMPLES


# -- bounded as the sources grow ------------------------------------------


def test_the_reference_prefix_does_not_grow_with_the_dataset(monkeypatch):
    """A hundred-fold larger dataset must not mean a larger prompt."""
    small = reference.examples()
    big = [
        ([f"Question {n} about brands?", "Brand A | 1", "Brand B | 2"], f"Example {n}")
        for n in range(500)
    ] + list(small)

    before = len(reference.prompt_prefix(GENDER))
    monkeypatch.setattr(reference, "_cache", big)
    after = len(reference.prompt_prefix(GENDER))

    assert after <= before * 2, f"prefix grew from {before} to {after} chars"
    assert reference.prompt_prefix(GENDER).count("Correct answer:") <= (
        reference.MAX_REFERENCE_EXAMPLES
    )


def test_the_correction_prefix_does_not_grow_with_the_library(tmp_path):
    memory = CorrectionMemory(persist=False)
    memory.use_document("doc.docx")

    for number in range(CORRECTION_POOL + 30):
        memory.record(
            Correction(
                label=f"Q{number}",
                original_lines=[f"Question {number} about brands?", "Brand A | 1"],
                ai_said={"element": "radio"},
                sp_corrected_to={"element": "checkbox"},
            )
        )

    prefix = memory.prompt_prefix(GENDER)
    assert prefix.count("The survey programmer corrected") <= MAX_CORRECTIONS
    assert len(memory.recent()) == CORRECTION_POOL, "the pool itself stays bounded too"


def test_corrections_are_chosen_by_relevance_not_recency():
    memory = CorrectionMemory(persist=False)
    memory.use_document("doc.docx")

    memory.record(
        Correction(
            label="QGENDER",
            original_lines=GENDER,
            ai_said={"element": "checkbox"},
            sp_corrected_to={"element": "radio"},
        )
    )
    for number in range(5):
        memory.record(
            Correction(
                label=f"QLATER{number}",
                original_lines=["How many international flights did you take?"],
                ai_said={"element": "radio"},
                sp_corrected_to={"element": "number"},
            )
        )

    prefix = memory.prompt_prefix(GENDER)
    assert "gender identity" in prefix, "the relevant one beat five newer ones"


def test_without_a_current_question_the_most_recent_are_used():
    """The old behaviour, still there for a caller with nothing to match on."""
    memory = CorrectionMemory(persist=False)
    memory.use_document("doc.docx")
    for number in range(5):
        memory.record(
            Correction(
                label=f"Q{number}",
                original_lines=[f"Question {number}"],
                ai_said={"element": "radio"},
                sp_corrected_to={"element": "checkbox"},
            )
        )

    prefix = memory.prompt_prefix()
    assert prefix.count("The survey programmer corrected") <= MAX_CORRECTIONS
    assert "Question 4" in prefix


# -- both sources reach a real call ---------------------------------------


def test_a_live_call_carries_seed_examples_dataset_precedent_and_corrections():
    from app.classify.classifier import SYSTEM_PROMPT, classify_question
    from app.classify.corrections import correction_memory
    from app.classify.lines import SourceLine

    class Stub:
        def __init__(self):
            self.systems: list[str] = []

        def generate_json(self, system, prompt):
            self.systems.append(system)
            return {
                "element": "radio",
                "title_lines": [0],
                "option_lines": [1, 2, 3],
                "confidence": 0.9,
            }

    correction_memory.clear()
    correction_memory.use_document("doc.docx")
    correction_memory.record(
        Correction(
            label="QOLD",
            original_lines=GENDER,
            ai_said={"element": "checkbox"},
            sp_corrected_to={"element": "radio"},
        )
    )

    stub = Stub()
    lines = [SourceLine(index=i, text=text) for i, text in enumerate(GENDER)]
    classify_question("Q1", lines, stub)
    correction_memory.clear()

    system = stub.systems[0]
    assert "Worked examples" in system, "seed library"
    assert "precedent, not a rule" in system, "reference dataset"
    assert "The survey programmer corrected" in system, "correction library"
    assert system.endswith(SYSTEM_PROMPT)
