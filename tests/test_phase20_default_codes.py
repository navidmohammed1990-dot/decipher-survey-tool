"""Phase 20 — house default codes for the four special row categories.

    91  Other (please specify)
    97  Don't know / Not sure
    98  Prefer not to say
    99  None of the above

A code the source document gives always wins. These apply only where it gives
none — the priority rule itself is unchanged from Bug 3 / Phase 10.
"""

from __future__ import annotations

import pytest

from app.generate.labels import default_code, is_opt_out, label_rows
from app.generate.xml_generator import generate_question
from app.models.survey import OptionLine, Question, TextRun

ROW_ELEMENTS = ("radio", "checkbox", "select", "radio_grid", "checkbox_grid", "number")


def labels_for(options: list[OptionLine], element: str = "radio") -> dict[str, str]:
    return {line.option.raw_text: line.label for line in label_rows(options, element=element)}


def question(element: str, options: list[OptionLine]) -> Question:
    is_grid = element in ("radio_grid", "checkbox_grid")
    return Question(
        label="Q1",
        element=element,
        title=[TextRun(text="Pick one")],
        options=[] if is_grid else options,
        rows=options if is_grid else [],
        cols=[OptionLine(raw_text="A"), OptionLine(raw_text="B")] if is_grid else [],
        needs_review=False,
    )


# -- detection -------------------------------------------------------------


@pytest.mark.parametrize(
    "text, suffix",
    [
        ("Other (please specify)", 91),
        ("Other, please specify", 91),
        ("Don't know", 97),
        ("Dont know", 97),
        ("Do not know", 97),
        ("Not sure", 97),
        ("Prefer not to say", 98),
        ("Prefer not to answer", 98),
        ("None of the above", 99),
        ("None of these", 99),
    ],
)
def test_each_category_has_its_house_code(text, suffix):
    assert default_code(text) == suffix


@pytest.mark.parametrize(
    "text",
    [
        "I don't know how often I shop there",
        "We do not know our exact revenue yet",
        "Other brands I would rather not say much about",
        "Reliable parcel delivery",
        "Not sure why the parcel was late, but it arrived",
        "Don't know the exact number of parcels",
        "Prefer not to say which brand I bought",
    ],
)
def test_the_phrases_inside_a_longer_answer_are_not_a_category(text):
    """Opening with the words is not enough - they must be all the row says."""
    assert default_code(text) is None


@pytest.mark.parametrize(
    "text, suffix",
    [
        ("Don't know / Not sure", 97),
        ("Don't know / Can't say", 97),
        ("Don't know or prefer not to say", 97),
        ("Unsure", 97),
    ],
)
def test_two_opt_out_phrases_joined_are_still_one_category(text, suffix):
    assert default_code(text) == suffix


# -- the defaults, and the rule they defer to ------------------------------


def test_dont_know_with_no_source_code_defaults_to_97():
    options = [OptionLine(raw_text="Option 1"), OptionLine(raw_text="Don't know")]
    assert labels_for(options)["Don't know"] == "r97"


def test_prefer_not_to_say_with_no_source_code_defaults_to_98():
    options = [OptionLine(raw_text="Option 1"), OptionLine(raw_text="Prefer not to say")]
    assert labels_for(options)["Prefer not to say"] == "r98"


def test_an_explicit_source_code_still_wins():
    """A9b codes "Don't know" as 98; that is the document's call, not ours."""
    options = [
        OptionLine(raw_text="Held by the 3PL provider", code="1"),
        OptionLine(raw_text="Don't know", code="98"),
    ]
    assert labels_for(options)["Don't know"] == "r98"


def test_all_four_categories_with_mixed_explicit_and_missing_codes():
    """The case that actually proves the priority order."""
    options = [
        OptionLine(raw_text="Option 1"),
        OptionLine(raw_text="Option 2", code="7"),
        OptionLine(raw_text="Other (please specify)", code="5"),
        OptionLine(raw_text="Don't know"),
        OptionLine(raw_text="Prefer not to say"),
        OptionLine(raw_text="None of the above"),
    ]
    assert labels_for(options) == {
        "Option 1": "r1",
        "Option 2": "r7",
        "Other (please specify)": "r5",
        "Don't know": "r97",
        "Prefer not to say": "r98",
        "None of the above": "r99",
    }


def test_a_house_code_the_source_spent_elsewhere_is_not_reused():
    """Two rows with one label is worse than a special row numbered plainly."""
    options = [
        OptionLine(raw_text="Option 1"),
        OptionLine(raw_text="Don't know", code="98"),
        OptionLine(raw_text="Prefer not to say"),
    ]
    labels = labels_for(options)

    assert labels["Don't know"] == "r98"
    assert labels["Prefer not to say"] != "r98"
    assert len(set(labels.values())) == len(labels), "labels must be unique"


def test_two_rows_of_the_same_category_do_not_collide():
    options = [OptionLine(raw_text="Don't know"), OptionLine(raw_text="Not sure")]
    labels = labels_for(options)

    assert labels["Don't know"] == "r97"
    assert len(set(labels.values())) == 2


def test_the_sequential_counter_still_skips_only_what_it_must():
    """[A, Other specify, B] stays r1, r91, r2 — the special row costs nothing."""
    options = [
        OptionLine(raw_text="A"),
        OptionLine(raw_text="Other (please specify)"),
        OptionLine(raw_text="B"),
    ]
    assert [line.label for line in label_rows(options, element="radio")] == [
        "r1",
        "r91",
        "r2",
    ]


# -- consistency across every element that generates rows ------------------


@pytest.mark.parametrize("element", ROW_ELEMENTS)
def test_the_defaults_apply_to_every_row_generating_element(element):
    options = [
        OptionLine(raw_text="Option 1"),
        OptionLine(raw_text="Other (please specify)"),
        OptionLine(raw_text="Don't know"),
        OptionLine(raw_text="Prefer not to say"),
        OptionLine(raw_text="None of the above"),
    ]
    xml = generate_question(question(element, options))

    for label in ("r1", "r91", "r97", "r98", "r99"):
        assert f'label="{label}"' in xml, f"{label} missing from {element}"


def test_a_numeric_grids_dont_know_is_a_noanswer_with_the_house_code():
    options = [
        OptionLine(raw_text="For Leisure", code="1", min_value="0", max_value="200"),
        OptionLine(raw_text="Don't know"),
    ]
    xml = generate_question(question("number", options))

    assert '<noanswer label="r97">Don\'t know</noanswer>' in xml
    assert '<row label="r1">For Leisure</row>' in xml


def test_opt_out_covers_more_than_the_coded_categories():
    """"Not applicable" is an opt-out with no house code of its own."""
    assert is_opt_out("Not applicable")
    assert default_code("Not applicable") is None


# -- no regression on the two categories that already worked ---------------


def test_other_specify_keeps_its_open_box():
    xml = generate_question(
        question("radio", [OptionLine(raw_text="Other (please specify)")])
    )
    assert '<row label="r91" value="91" open="1" openSize="25" randomize="0">' in xml


@pytest.mark.parametrize("element, exclusive", [("radio", False), ("checkbox", True)])
def test_none_of_the_above_keeps_its_attributes(element, exclusive):
    xml = generate_question(
        question(element, [OptionLine(raw_text="None of the above")])
    )
    row = next(line for line in xml.splitlines() if "None of the above" in line)

    assert 'label="r99"' in row
    assert 'randomize="0"' in row
    assert ('exclusive="1"' in row) is exclusive


# -- Phase 20b: randomize, and the same gate on the older two categories ---


@pytest.mark.parametrize("element", ["radio", "checkbox", "select", "radio_grid", "checkbox_grid"])
def test_the_opt_out_rows_are_fixed_at_the_end_of_the_list(element):
    """Confirmed in 20b: they behave like None, not like an ordinary option."""
    options = [
        OptionLine(raw_text="Option 1"),
        OptionLine(raw_text="Don't know"),
        OptionLine(raw_text="Prefer not to say"),
        OptionLine(raw_text="None of the above"),
    ]
    xml = generate_question(question(element, options))

    for label in ("r97", "r98", "r99"):
        row = next(line for line in xml.splitlines() if f'label="{label}"' in line)
        assert 'randomize="0"' in row, f"{label} should not randomise in {element}"

    plain = next(line for line in xml.splitlines() if 'label="r1"' in line)
    assert "randomize=" not in plain, "an ordinary option still randomises"


def test_a_numeric_grids_noanswer_stays_bare():
    """20b asks for no change here - there is no evidence either way."""
    options = [
        OptionLine(raw_text="For Leisure", code="1", min_value="0", max_value="200"),
        OptionLine(raw_text="Don't know"),
    ]
    xml = generate_question(question("number", options))

    assert '<noanswer label="r97">Don\'t know</noanswer>' in xml


def test_none_of_the_above_inside_a_real_answer_is_not_the_none_row():
    """The case named in the brief."""
    text = "None of the above brands appeal to me"
    assert default_code(text) is None

    xml = generate_question(
        question("checkbox", [OptionLine(raw_text="Option 1"), OptionLine(raw_text=text)])
    )
    row = next(line for line in xml.splitlines() if "brands appeal" in line)

    assert 'label="r99"' not in row
    assert "exclusive=" not in row, "a real answer must not be marked exclusive"
    assert "randomize=" not in row


@pytest.mark.parametrize(
    "text",
    [
        "Please specify any other brands you have used",
        "Other supermarkets I shop at, please specify which ones and how often",
        "None of the above stores are near my home",
    ],
)
def test_the_older_two_categories_now_take_the_same_gate(text):
    assert default_code(text) is None


@pytest.mark.parametrize(
    "text, suffix",
    [
        ("Other (please specify)", 91),
        ("Other, please specify", 91),
        ("Other - please specify", 91),
        ("Other (specify)", 91),
        ("Other (please specify below)", 91),
        ("None of the above", 99),
        ("None of these", 99),
        ("None of these apply", 99),
        ("None of the above apply to me", 99),
    ],
)
def test_the_genuine_cases_survive_the_tightening(text, suffix):
    assert default_code(text) == suffix


def test_an_other_specify_row_keeps_its_open_box_after_tightening():
    xml = generate_question(
        question("radio", [OptionLine(raw_text="Other (please specify)")])
    )
    assert 'open="1"' in xml and 'openSize="25"' in xml
