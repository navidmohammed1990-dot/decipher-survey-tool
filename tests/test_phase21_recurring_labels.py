"""Phase 21 — a repeated prefix is a label style, whatever it spells.

Phase 15 made a blank line a segmentation signal in its own right. That fixed
the Sent1/Sent1B/Sent2 paste as long as blank lines separated the questions;
pasted without them the same questionnaire collapsed back into one question.

The answer is not a wider letter bound - that only moves the wall to the next
house style. A style announces itself by repeating.
"""

from __future__ import annotations

import pytest

from app.classify.paste import recurring_prefix, paragraph_blocks, split_questions

NO_GAPS = (
    "Sent1. How much do you agree with this statement?\n"
    "Strongly agree | 1\n"
    "Agree | 2\n"
    "Sent1B. And how confident are you in that answer?\n"
    "Very confident | 1\n"
    "Sent2. Which of these brands do you recall?\n"
    "Brand A | 1\n"
    "Sent3. Which three areas matter most?\n"
    "Area A | 1\n"
    "Sent4. How likely are you to recommend?\n"
    "Very likely | 1\n"
    "Sent5. Why do you say that?\n"
    "Sent6. Anything else?\n"
)


def labels(text: str) -> list[str]:
    return [block.label for block in split_questions(text)[0]]


def prefix_of(text: str) -> str | None:
    return recurring_prefix(paragraph_blocks(text))


# -- the reported failure --------------------------------------------------


def test_an_unknown_prefix_with_no_blank_lines_still_splits():
    """Seven questions, not one 18-line block."""
    blocks, warnings = split_questions(NO_GAPS)

    assert [block.label for block in blocks] == [
        "SENT1",
        "SENT1B",
        "SENT2",
        "SENT3",
        "SENT4",
        "SENT5",
        "SENT6",
    ]
    assert any("label style" in w for w in warnings)


def test_the_real_labels_survive_rather_than_placeholders():
    blocks, _ = split_questions(NO_GAPS)
    assert not any(block.synthesised_label for block in blocks)


def test_each_question_keeps_only_its_own_options():
    blocks, _ = split_questions(NO_GAPS)
    first = [line.text for line in blocks[0].lines]

    assert "Agree | 2" in first
    assert not any("confident" in text for text in first)


@pytest.mark.parametrize("prefix", ["Sent", "Screener", "Awareness", "Zz"])
def test_any_prefix_works_because_none_is_enumerated(prefix):
    paste = (
        f"{prefix}1. First question?\nYes | 1\n"
        f"{prefix}2. Second question?\nNo | 1\n"
        f"{prefix}3. Third question?\nMaybe | 1\n"
    )
    assert labels(paste) == [f"{prefix.upper()}{n}" for n in (1, 2, 3)]


# -- when a repeated token is not a label style ---------------------------


def test_a_known_label_style_is_never_second_guessed():
    """The ordinary path found labels, so recurrence is not consulted."""
    paste = "Q1. First\nYes | 1\nQ2. Second\nNo | 1\nQ3. Third\nMaybe | 1\n"
    assert labels(paste) == ["Q1", "Q2", "Q3"]


def test_coded_options_are_not_label_candidates():
    """"Brand1 | 1" repeats a prefix but is an answer, not a heading."""
    paste = (
        "Which of these have you bought?\n"
        "Brand1. Coca-Cola | 1\n"
        "Brand2. Pepsi | 2\n"
        "Brand3. Fanta | 3\n"
    )
    assert prefix_of(paste) is None
    assert len(split_questions(paste)[0]) == 1


def test_a_prefix_appearing_once_is_not_a_style():
    paste = "Intro1. Welcome to the survey\nPlease continue\n"
    assert prefix_of(paste) is None


def test_a_token_with_no_separator_is_not_a_label():
    """"Sent1 How much" without punctuation stays one question."""
    paste = "Sent1 How much do you agree?\nYes\nSent2 And how confident?\nNo\n"
    assert prefix_of(paste) is None


def test_plain_numbering_still_wins_when_nothing_recurs():
    blocks, warnings = split_questions(
        "1. First question\nYes\nNo\n2. Second question\nYes\nNo"
    )
    assert [block.label for block in blocks] == ["1", "2"]
    assert any("plain numbering" in w for w in warnings)


def test_a_recurring_prefix_outranks_plain_numbering():
    """"Sent1." recurring says more about the document than a stray "1." does."""
    paste = (
        "Sent1. First question?\n1. Yes\n2. No\n"
        "Sent2. Second question?\n1. Yes\n2. No\n"
    )
    assert labels(paste) == ["SENT1", "SENT2"]


def test_an_unlabelled_paste_is_untouched():
    blocks, warnings = split_questions(
        "Which of these have you used?\nEmail\nText message"
    )
    assert len(blocks) == 1
    assert any("whole paste as one question" in w for w in warnings)


def test_gap_segmentation_still_applies_when_nothing_recurs():
    paste = (
        "Which of these have you used in the past month?\nEmail\n"
        "\n"
        "And which do you prefer?\nEmail\n"
    )
    assert labels(paste) == ["Q1", "Q2"]


# -- the DOCX path shares the judgment ------------------------------------


def test_a_docx_with_an_unknown_prefix_and_no_gaps_still_splits():
    """Without this the whole file became one preamble and nothing classified."""
    from app.models.document import ParagraphBlock
    from app.parsing.question_boundaries import detect_boundaries

    lines = [
        "Sent1. How much do you agree?",
        "Strongly agree",
        "Sent2. Which brands do you recall?",
        "Brand A",
        "Sent3. Why do you say that?",
    ]
    blocks = [ParagraphBlock(index=i, text=t) for i, t in enumerate(lines)]

    boundaries, warnings = detect_boundaries(blocks)

    assert [b.label for b in boundaries] == ["SENT1", "SENT2", "SENT3"]
    assert not any(b.is_preamble for b in boundaries)
    assert any("label style" in w for w in warnings)


def test_a_docx_with_known_labels_is_unaffected():
    from app.models.document import ParagraphBlock
    from app.parsing.question_boundaries import detect_boundaries

    lines = ["Q1. Pick one", "Yes", "", "Q2. Pick another", "No"]
    blocks = [ParagraphBlock(index=i, text=t) for i, t in enumerate(lines)]

    boundaries, warnings = detect_boundaries(blocks)

    assert [b.label for b in boundaries] == ["Q1", "Q2"]
    assert not warnings
