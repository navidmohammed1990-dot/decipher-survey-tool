"""Phase 15: segmentation and grid detection judged, not pattern-matched.

The wording here is deliberately unlike anything in the seed dataset or the
correction library. A test that reuses a seed example only re-confirms the
example; these are meant to fail if the code has learned phrases instead of
principles.
"""

from __future__ import annotations

import pytest

from app.classify.features import detect_type_tag
from app.classify.paste import join_cells, split_questions
from app.models.document import ParagraphBlock
from app.parsing.question_boundaries import detect_boundaries


def labels(text: str) -> list[str]:
    return [q.label for q in split_questions(text)[0]]


def texts(text: str) -> list[list[str]]:
    return [[line.text for line in q.lines] for q in split_questions(text)[0]]


# -- A: blank lines are a segmentation signal in their own right ----------


def test_a_label_style_the_regex_cannot_read_still_splits():
    """Sent1/Sent1B/Sent2 - four letters, outside the label shape's bound."""
    paste = (
        "Sent1. How much do you agree with this statement?\n"
        "Strongly agree | 1\n"
        "Agree | 2\n"
        "\n"
        "Sent1B. And how confident are you in that answer?\n"
        "Very confident | 1\n"
        "Somewhat confident | 2\n"
        "\n"
        "Sent2. Which of these brands do you recall?\n"
        "Brand A | 1\n"
        "Brand B | 2\n"
    )
    blocks, warnings = split_questions(paste)

    assert len(blocks) == 3, "blank lines separated three questions"
    # Phase 21 improved this: the repeated "Sent" prefix is now read as this
    # document's label style, so the real labels survive instead of
    # placeholders. The gap split still decides where the questions are.
    assert [block.label for block in blocks] == ["SENT1", "SENT1B", "SENT2"]
    assert not any(block.synthesised_label for block in blocks)
    assert any("label style" in w for w in warnings)


def test_a_options_do_not_bleed_between_gap_split_questions():
    paste = (
        "Item1. Pick a colour\nRed | 1\nBlue | 2\n"
        "\n"
        "Item2. Pick a size\nSmall | 1\nLarge | 2\n"
    )
    first, second = texts(paste)

    assert "Small" not in " ".join(first)
    assert "Red" not in " ".join(second)


def test_a_gap_split_questions_each_restart_their_codes():
    """The duplicate-row-label bug: two questions' options in one block."""
    paste = "Stmt1. Agree?\nYes | 1\nNo | 2\n\nStmt2. Sure?\nYes | 1\nNo | 2\n"
    blocks, _ = split_questions(paste)

    for block in blocks:
        codes = [line.features.trailing_numeric_code for line in block.lines[1:]]
        assert codes == ["1", "2"]


def test_a_a_paste_with_no_labels_anywhere_splits_on_gaps_alone():
    """No label pattern to lean on - the gaps are the only signal there is."""
    paste = (
        "Which of these have you used in the past month?\nEmail\nText message\n"
        "\n"
        "And which do you prefer?\nEmail\nText message\n"
    )
    assert labels(paste) == ["Q1", "Q2"]


def test_a_one_unbroken_paragraph_is_still_one_question():
    """The fallback must survive: no gaps, no label, one selection."""
    blocks, warnings = split_questions(
        "Roughly how often do you order takeaway?\nWeekly\nMonthly\nRarely"
    )

    assert len(blocks) == 1
    assert blocks[0].label == "Q1"
    assert any("whole paste as one question" in w for w in warnings)


def test_a_labels_still_split_without_any_blank_line():
    """Gaps are an additional signal, not a replacement for labels."""
    paste = "QZ5. Pick one\nYes | 1\nNo | 2\nQZ6. Pick another\nYes | 1\nNo | 2"
    assert labels(paste) == ["QZ5", "QZ6"]


def test_a_two_labels_inside_one_gap_block_still_split():
    paste = "P1. First\nYes | 1\nP2. Second\nNo | 1"
    assert labels(paste) == ["P1", "P2"]


def test_a_a_coded_answer_list_under_a_gap_joins_the_question_above():
    paste = "MP2. Which do you own?\n\nA phone | 1\nA tablet | 2\nNeither | 99\n"
    blocks, _ = split_questions(paste)

    assert len(blocks) == 1
    assert [line.text for line in blocks[0].lines] == [
        "Which do you own?",
        "A phone | 1",
        "A tablet | 2",
        "Neither | 99",
    ]


def test_a_an_unlabelled_block_in_a_labelled_paste_is_a_continuation():
    """A labelled paste means an unlabelled block continues, not restarts."""
    paste = "QD24. Rate the following\n\nVery good | 1\nPoor | 2\n\nThe staff\nThe app\n"
    blocks, warnings = split_questions(paste)

    assert len(blocks) == 1
    assert blocks[0].label == "QD24"
    assert "The app" in [line.text for line in blocks[0].lines]
    assert any("attached to the question above" in w for w in warnings)


def test_a_docx_with_unreadable_labels_splits_on_empty_paragraphs():
    """The DOCX path shared the bug: everything became one preamble."""
    lines = [
        "Sent1. How much do you agree?",
        "Strongly agree",
        "",
        "Sent2. Which brands do you recall?",
        "Brand A",
    ]
    blocks = [ParagraphBlock(index=i, text=t) for i, t in enumerate(lines)]

    boundaries, warnings = detect_boundaries(blocks)

    # Phase 21 improved this too: the repeated "Sent" prefix is recognised as
    # the document's label style, so the real labels survive. The gap fallback
    # below still covers a document where nothing recurs.
    assert [b.label for b in boundaries] == ["SENT1", "SENT2"]
    assert not any(b.is_preamble for b in boundaries)
    assert any("label style" in w for w in warnings)


def test_a_docx_with_no_recurring_prefix_still_splits_on_empty_paragraphs():
    lines = [
        "How much do you agree?",
        "Strongly agree",
        "",
        "Which brands do you recall?",
        "Brand A",
    ]
    blocks = [ParagraphBlock(index=i, text=t) for i, t in enumerate(lines)]

    boundaries, warnings = detect_boundaries(blocks)

    assert [b.label for b in boundaries] == ["Q1", "Q2"]
    assert any("split on blank lines" in w for w in warnings)


def test_a_docx_readable_labels_are_unaffected_by_the_fallback():
    lines = ["Q1. Pick one", "Yes", "", "Q2. Pick another", "No"]
    blocks = [ParagraphBlock(index=i, text=t) for i, t in enumerate(lines)]

    boundaries, _ = detect_boundaries(blocks)
    assert [b.label for b in boundaries] == ["Q1", "Q2"]


# -- B: a grid is a concept, not a phrase --------------------------------


@pytest.mark.parametrize(
    "marker, expected",
    [
        ("SR PER ROW", "SR_GRID"),
        ("SR per statement", "SR_GRID"),
        ("MR per row", "MR_GRID"),
        ("MR per brand", "MR_GRID"),
        ("SR for each item", "SR_GRID"),
        ("SR against each attribute", "SR_GRID"),
        ("MR PER STATEMENT", "MR_GRID"),
        ("one response per attribute", "GRID"),
        ("single answer for each brand", "GRID"),
    ],
)
def test_b_grid_wording_is_recognised_by_shape(marker, expected):
    assert detect_type_tag(marker) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Mr. Smith visits per week",
        "I pay $5 per month",
        "Once per day",
        "How many times per week do you shop?",
    ],
)
def test_b_ordinary_prose_containing_per_is_not_a_grid(text):
    assert detect_type_tag(text) not in {"SR_GRID", "MR_GRID", "GRID"}


def test_b_sr_stays_case_sensitive_so_titles_are_not_markers():
    assert detect_type_tag("Mr per row") is None


SCALE = ["Strongly Disagree | 1", "Disagree | 2", "Neither | 3", "Agree | 4"]


def test_b_space_aligned_columns_become_columns():
    """The APP3 failure: a space-aligned scale collapsed into one line."""
    paste = (
        "APP3. How much do you agree with each of the following?  SR per statement\n"
        "\n"
        "     Strongly Disagree   Disagree   Neither   Agree\n"
        "             1               2         3        4\n"
        "The service was easy to use\n"
        "The price was fair\n"
    )
    blocks, _ = split_questions(paste)

    assert len(blocks) == 1, "the table belongs to APP3, not to a new question"
    assert blocks[0].label == "APP3"
    assert blocks[0].lines[0].features.type_tag_value == "SR_GRID"
    assert [line.text for line in blocks[0].lines[1:5]] == [
        "Strongly Disagree | 1",
        "Disagree | 2",
        "Neither | 3",
        "Agree | 4",
    ]
    assert [line.features.trailing_numeric_code for line in blocks[0].lines[1:5]] == [
        "1",
        "2",
        "3",
        "4",
    ]


def test_b_layout_does_not_change_the_result():
    """Space, tab and pipe separated scales must all parse identically."""
    stem = "APP3. Rate each of these  SR per statement\n"
    rows = "The service was easy to use\nThe price was fair\n"
    variants = {
        "space": "     Strongly Disagree   Disagree   Neither   Agree\n"
        "             1               2         3        4\n",
        "tab": "\tStrongly Disagree\tDisagree\tNeither\tAgree\n\t1\t2\t3\t4\n",
        "pipe": "Strongly Disagree | Disagree | Neither | Agree\n1 | 2 | 3 | 4\n",
    }

    results = {
        name: [line.text for line in split_questions(stem + table + rows)[0][0].lines]
        for name, table in variants.items()
    }

    assert results["space"] == results["tab"] == results["pipe"]
    assert results["space"][1:5] == [
        "Strongly Disagree | 1",
        "Disagree | 2",
        "Neither | 3",
        "Agree | 4",
    ]


def test_b_a_two_column_option_row_is_not_treated_as_a_scale():
    """The column splitter needs three columns; codes must survive."""
    assert join_cells("Under 18 years        1") == "Under 18 years | 1"
    assert join_cells("I have lived here 20 years") == "I have lived here 20 years"


def test_b_a_sentence_with_a_double_space_is_left_alone():
    assert join_cells("Yes, always.  No, never.") == "Yes, always.  No, never."


# -- audit: the same gap found elsewhere ---------------------------------


@pytest.mark.parametrize("header", ["ASK ALL, SC", "[MC]", "SC:", "SR", "MR", "[GRID]"])
def test_audit_known_headers_still_move_to_the_next_question(header):
    from app.parsing.question_boundaries import is_question_header

    assert is_question_header(header)


@pytest.mark.parametrize("header", ["SR PER ROW", "SR per statement", "MR per brand"])
def test_audit_grid_headers_are_no_longer_stranded(header):
    """The boundary detector kept its own marker list; SR/MR never reached it."""
    from app.parsing.question_boundaries import is_question_header

    assert is_question_header(header)


@pytest.mark.parametrize(
    "line",
    [
        "TERMINATE IF Q1=2",
        "How much do you agree? SC",
        "Please select all that apply",
        "Randomise the list",
    ],
)
def test_audit_a_line_that_is_not_only_a_marker_stays_put(line):
    from app.parsing.question_boundaries import is_question_header

    assert not is_question_header(line)


def test_audit_a_grid_header_moves_onto_its_own_question():
    from app.models.document import ParagraphBlock

    lines = ["Q1. Pick one", "Yes", "SR PER ROW", "Q2. Rate these", "The staff"]
    blocks = [ParagraphBlock(index=i, text=t) for i, t in enumerate(lines)]

    first, second = detect_boundaries(blocks)[0]
    assert first.block_indices == [0, 1]
    assert second.block_indices == [2, 3, 4], "the marker introduces Q2"


def test_audit_a_narrow_column_wrap_merges():
    """A fixed 25-character floor missed every wrap in a narrow table column."""
    paste = (
        "Q1. Which would you do?\n"
        "Buy it instead of\nanother brand | 1\n"
        "Wait for the price\nto come down | 2\n"
        "Neither of these | 3\n"
    )
    assert texts(paste)[0] == [
        "Which would you do?",
        "Buy it instead of another brand | 1",
        "Wait for the price to come down | 2",
        "Neither of these | 3",
    ]


def test_audit_a_variable_name_is_still_not_half_an_option():
    paste = (
        "Q2. Please create a derived variable\n"
        "S2_AGE BANDS\n"
        "18 to 34 | 1\n"
        "35 to 54 | 2\n"
        "55 or over | 3\n"
    )
    assert "S2_AGE BANDS" in texts(paste)[0]
