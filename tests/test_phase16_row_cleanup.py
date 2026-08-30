"""Row cleanups the 27-entry dataset exposed.

Three deterministic reflow fixes: a wrapped title must not swallow the routing
line below it, a directive written inside an option's own cell is not part of
the option, and a leading row-id column is not part of the row's wording.

None of these decides a role. Each only changes where one line ends and the
next begins, which stays deterministic by design.
"""

from __future__ import annotations

import pytest

from app.classify.paste import join_cells, split_questions, strip_leading_id_column


def texts(text: str) -> list[str]:
    return [line.text for line in split_questions(text)[0][0].lines]


# -- 1. a wrapped title must not swallow the line below it ----------------


def test_a_wrapped_title_does_not_absorb_the_routing_line():
    """The routing line carried a code, so it looked like a wrap completion."""
    paste = (
        "Sent3. Which three areas are most important for Australia Post to get\n"
        "right in your community over the next 4 years? (Please select 3)\n"
        "RANDOMISE\tMR, SELECT 3\n"
        "Reliable parcel delivery\t1\n"
        "Faster parcel delivery\t2\n"
        "Affordability\t3\n"
    )
    lines = texts(paste)

    assert "right in your community over the next 4 years? (Please select 3)" in lines
    assert "RANDOMISE | MR, SELECT 3" in lines
    assert not any("RANDOMISE" in line and "community" in line for line in lines)


def test_a_wrapped_option_still_merges():
    """The guard must not cost us the wrap merging it sits next to."""
    paste = (
        "QD24. If you were to purchase this product, would you...?\n"
        "Buy it instead of another [BRAND] [FORMAT OF INTEREST]\n"
        "product you usually buy 1\n"
        "Buy it instead of a different type of product    3\n"
    )
    assert (
        "Buy it instead of another [BRAND] [FORMAT OF INTEREST] "
        "product you usually buy | 1"
    ) in texts(paste)


def test_a_type_marker_line_is_not_a_wrap_completion():
    paste = (
        "Q1. Which of the following describes how you feel about the service\n"
        "your provider gave you over the last twelve months?\n"
        "ASK ALL, SC 1\n"
        "Very good\t1\n"
        "Quite good\t2\n"
        "Poor\t3\n"
    )
    assert not any(
        "provider gave you" in line and "ASK ALL" in line for line in texts(paste)
    )


# -- 2. a directive inside the option's own cell --------------------------


def test_a_directive_in_the_option_cell_becomes_a_row_note():
    assert (
        join_cells("Other (specify) OE, ANCHOR BASE\t98")
        == "Other (specify) | 98 | OE, ANCHOR BASE"
    )


def test_the_same_split_works_on_space_aligned_columns():
    assert (
        join_cells("Other (specify) OE, ANCHOR BASE   98")
        == "Other (specify) | 98 | OE, ANCHOR BASE"
    )


@pytest.mark.parametrize(
    "line, expected",
    [
        ("Other (specify)\t98", "Other (specify) | 98"),
        ("Male\t1", "Male | 1"),
        ("Yes  1", "Yes | 1"),
        ("None of the above\t99", "None of the above | 99"),
    ],
)
def test_an_option_with_no_directive_is_unchanged(line, expected):
    assert join_cells(line) == expected


def test_a_cell_that_is_only_a_marker_is_left_alone():
    """A bare marker is a type signal, not an option with a note."""
    assert join_cells("MR, SELECT 3") == "MR, SELECT 3"


def test_a_title_carrying_a_marker_is_not_split():
    assert join_cells("How much do you agree? SC") == "How much do you agree? SC"


# -- 3. a leading row-id column -------------------------------------------


@pytest.mark.parametrize("gap", ["   ", "\t"])
def test_a_leading_id_column_is_dropped(gap):
    """Layout must not matter here either - spaces and tabs both mark columns."""
    paste = (
        "GL1d. Which of these apply?\n"
        + f"01{gap}Has the lowest prices{gap}1\n"
        + f"02{gap}Offers a large selection{gap}2\n"
        + f"03{gap}Offers great customer service{gap}3\n"
    )
    assert texts(paste)[1:] == [
        "Has the lowest prices | 1",
        "Offers a large selection | 2",
        "Offers great customer service | 3",
    ]


def test_a_band_list_keeps_the_number_it_starts_with():
    """"18 to 24 years" opens with a number but 18 is not its code."""
    paste = "Q1. How old are you?\n18 to 24 years   1\n25 to 34 years   2\n35 to 44 years   3\n"
    assert texts(paste)[1:] == [
        "18 to 24 years | 1",
        "25 to 34 years | 2",
        "35 to 44 years | 3",
    ]


def test_a_number_that_is_the_option_text_survives():
    """"1 year | 1" matches the code, but the 1 is not a separate column."""
    paste = "Q1. How many years?\n1 year   1\n2 years   2\n3 years   3\n"
    assert texts(paste)[1:] == ["1 year | 1", "2 years | 2", "3 years | 3"]


def test_ids_that_do_not_repeat_the_code_are_left_alone():
    """Retired rows renumber the codes; stripping on shape alone would guess."""
    paste = "Q1. Rate these\n01  Alpha   17\n02  Beta   18\n03  Gamma   19\n"
    assert texts(paste)[1:] == ["01 Alpha | 17", "02 Beta | 18", "03 Gamma | 19"]


def test_two_agreeing_rows_are_not_enough():
    paste = "Q1. Pick\n01  Alpha   1\n02  Beta   2\n"
    assert texts(paste)[1:] == ["01 Alpha | 1", "02 Beta | 2"]


def test_one_disagreeing_row_leaves_the_whole_block_alone():
    lines = ["01  Alpha   1", "02  Beta   2", "03  Gamma   9"]
    assert strip_leading_id_column(lines) == lines
