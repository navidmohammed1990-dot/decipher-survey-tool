"""Phase 19 — numeric grids with per-row min/max constraints.

The pattern this exists for puts the row's code on the *left*, lets long rows
wrap over two or three lines, and states each row's numeric range as prose in
its own column. None of the existing machinery could see it: the wrap merger
needs a trailing code to close a pair, and there was no field to carry a range.
"""

from __future__ import annotations

import pytest

from app.classify.features import detect_numeric_bounds
from app.classify.paste import code_column_rows, split_questions
from app.dataset import build_question, load_examples
from app.generate.xml_generator import generate_question
from app.models.survey import OptionLine, Question, TextRun

CATHAY = (
    "ONUM\n"
    "How many international flights have you taken from [MARKET] in\n"
    "the last 12 months?\n"
    "If you took a return trip, please consider this as two (2) flights.\n"
    "R\n"
    "_1  For Leisure (i.e. holidays or short break, visiting family\n"
    "    and friends)     Open numeric response box; min 0 max 200\n"
    "_2  For Business (i.e. conferences, client presentations,\n"
    "    business development, client meetings)\n"
    "                     Open numeric response box; min 0 max 200\n"
    "991 None, I have not taken any international flights from\n"
    "    [MARKET] in the last 12 months\n"
)


def options_of(text: str) -> list[OptionLine]:
    lines = split_questions(text)[0][0].lines
    return [OptionLine.from_text(line.text) for line in lines]


# -- reading a range out of prose ----------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Open numeric response box; min 0 max 200", ("0", "200")),
        ("Min: 0, Max: 200", ("0", "200")),
        ("minimum 0 maximum 50", ("0", "50")),
        ("min 1 max 99", ("1", "99")),
    ],
)
def test_a_stated_range_is_read_whatever_the_wording(text, expected):
    assert detect_numeric_bounds(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "max 200",
        "TERMINATE",
        "I have 3 min and 4 max children",
        "Open numeric response box",
    ],
)
def test_half_a_range_or_none_at_all_is_not_a_range(text):
    """An open-ended range is not a reason to invent a zero."""
    assert detect_numeric_bounds(text) is None


# -- the real source pattern ---------------------------------------------


def test_the_cathay_rows_parse_with_their_ranges():
    parsed = {option.raw_text: option for option in options_of(CATHAY)}

    leisure = parsed[
        "For Leisure (i.e. holidays or short break, visiting family and friends)"
    ]
    assert (leisure.code, leisure.min_value, leisure.max_value) == ("1", "0", "200")

    business = parsed[
        "For Business (i.e. conferences, client presentations, "
        "business development, client meetings)"
    ]
    assert (business.code, business.min_value, business.max_value) == ("2", "0", "200")


def test_the_none_row_is_a_row_without_a_range():
    """It is an exclusive option, not a numeric entry - both bounds stay None."""
    parsed = {option.raw_text: option for option in options_of(CATHAY)}
    none_row = parsed[
        "None, I have not taken any international flights from [MARKET] "
        "in the last 12 months"
    ]

    assert none_row.code == "991"
    assert none_row.min_value is None
    assert none_row.max_value is None


def test_a_row_wrapping_over_three_lines_is_joined():
    """_2's wording runs across two lines before its range on a third."""
    texts = [line.text for line in split_questions(CATHAY)[0][0].lines]

    assert any(
        "For Business (i.e. conferences, client presentations, "
        "business development, client meetings)" in text
        for text in texts
    )


def test_the_question_stem_is_not_swallowed_into_a_row():
    texts = [line.text for line in split_questions(CATHAY)[0][0].lines]

    assert "How many international flights have you taken from [MARKET] in" in texts
    assert "If you took a return trip, please consider this as two (2) flights." in texts


# -- when a code column is not a code column ------------------------------


def test_rows_with_codes_on_the_right_are_left_to_the_ordinary_path():
    lines = ["Male  1", "Female  2", "Other  97"]
    assert code_column_rows(lines) is None


def test_a_band_list_is_not_a_code_column():
    """"18 to 24 years" leads with a number, but it is prose, not a column."""
    lines = ["18 to 24 years", "25 to 34 years", "35 to 44 years"]
    assert code_column_rows(lines) is None


def test_numbers_that_are_the_wording_are_not_a_code_column():
    lines = ["1 year", "2 years", "3 years"]
    assert code_column_rows(lines) is None


def test_two_rows_are_not_enough():
    lines = ["_1  For Leisure", "_2  For Business"]
    assert code_column_rows(lines) is None


def test_zero_padded_codes_count_as_unambiguous():
    lines = ["01 Alpha", "02 Beta", "03 Gamma"]
    assert code_column_rows(lines) is not None


# -- generation ------------------------------------------------------------


def numeric_question() -> Question:
    return Question(
        label="Q20",
        element="number",
        title=[TextRun(text="How many flights?")],
        options=[
            OptionLine(raw_text="For Leisure", code="1", min_value="0", max_value="200"),
            OptionLine(raw_text="For Business", code="2", min_value="0", max_value="200"),
            OptionLine(raw_text="None of these", code="991"),
        ],
        needs_review=False,
    )


def test_the_range_is_carried_once_on_the_question():
    """Confirmed against real generated XML: verify=range(), not per-row."""
    xml = generate_question(numeric_question())

    assert 'verify="range(0,200)"' in xml
    assert '<row label="r1">For Leisure</row>' in xml
    assert '<row label="r2">For Business</row>' in xml


def test_no_row_carries_min_or_max_attributes():
    xml = generate_question(numeric_question())

    assert 'min="' not in xml
    assert 'max="' not in xml


def test_a_numeric_grid_declares_its_list_display():
    assert 'ss:listDisplay="0"' in generate_question(numeric_question())


def test_amount_is_never_emitted():
    """Confirmed unnecessary: the programmer adds it by hand where needed."""
    assert "amount=" not in generate_question(numeric_question())


def test_an_opt_out_row_is_a_noanswer_not_a_row():
    xml = generate_question(numeric_question())

    assert '<noanswer label="r991">None of these</noanswer>' in xml
    assert "<row" in xml, "the numeric rows are still rows"


def test_a_rangeless_row_is_the_opt_out_whatever_it_says():
    """Structure decides where it can - no phrase needs recognising."""
    question = numeric_question()
    question.options[2] = OptionLine(raw_text="Something else entirely", code="97")
    xml = generate_question(question)

    assert '<noanswer label="r97">Something else entirely</noanswer>' in xml


def test_wording_decides_only_when_no_row_states_a_range():
    question = Question(
        label="A9b",
        element="number",
        title=[TextRun(text="What proportion...?")],
        options=[
            OptionLine(raw_text="Held by the 3PL provider", code="1"),
            OptionLine(raw_text="Held by my business", code="2"),
            OptionLine(raw_text="Don't know", code="98"),
        ],
        needs_review=False,
    )
    xml = generate_question(question)

    assert '<row label="r1">Held by the 3PL provider</row>' in xml
    assert '<noanswer label="r98">Don\'t know</noanswer>' in xml
    assert "verify=" not in xml, "no row stated a range, so none is invented"
    assert "amount=" not in xml


def test_radio_exclusion_rows_are_unaffected():
    """The r99/exclusive convention is separate and already confirmed."""
    for element in ("radio", "checkbox"):
        xml = generate_question(
            Question(
                label="Q1",
                element=element,
                title=[TextRun(text="Pick one")],
                options=[
                    OptionLine(raw_text="Option 1"),
                    OptionLine(raw_text="None of the above"),
                ],
                needs_review=False,
            )
        )
        assert "<noanswer" not in xml, f"{element} must not use noanswer"
        assert '<row label="r99"' in xml
        assert 'randomize="0"' in xml


def test_a_single_value_number_question_still_has_no_rows():
    """Giving number rows must not add any to a question that has none."""
    xml = generate_question(
        Question(
            label="Q1",
            element="number",
            title=[TextRun(text="How old are you?")],
            needs_review=False,
        )
    )

    assert "<row" not in xml
    assert '<number label="Q1" size="3" optional="0">' in xml


def test_the_dataset_entry_generates_the_expected_shape():
    example = next(
        e for e in load_examples() if e.id == "number_grid_with_per_row_constraints"
    )
    xml = generate_question(build_question(example))

    assert xml.count("<row ") == 2
    assert xml.count("<noanswer ") == 1
    assert 'verify="range(0,200)"' in xml
    assert 'min="' not in xml and 'max="' not in xml
