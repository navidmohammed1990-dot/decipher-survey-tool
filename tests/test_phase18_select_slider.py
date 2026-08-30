"""Phase 18 — select_slider, the standalone rating scale.

The response set *is* the scale, so its points are <choice> elements rather
than <row>s, and a point that opts out of the scale is marked rather than
numbered along it.
"""

from __future__ import annotations

import pytest

from app.classify.classifier import SYSTEM_PROMPT
from app.dataset import build_question, load_examples
from app.generate.labels import label_choices
from app.generate.resources import resource_tag_for
from app.generate.xml_generator import generate_question
from app.models.survey import (
    CHOICE_ELEMENTS,
    OPTION_ELEMENTS,
    SUPPORTED_ELEMENTS,
    OptionLine,
    Question,
    TextRun,
)

SCALE = [OptionLine(raw_text=str(n), code=str(n)) for n in range(1, 6)]


def slider(options: list[OptionLine], label: str = "A20") -> Question:
    return Question(
        label=label,
        element="select_slider",
        title=[TextRun(text="How likely are you to remember this ad?")],
        options=options,
        needs_review=False,
    )


# -- the element is registered everywhere ---------------------------------


def test_the_element_is_supported():
    assert "select_slider" in SUPPORTED_ELEMENTS
    assert "select_slider" in OPTION_ELEMENTS
    assert "select_slider" in CHOICE_ELEMENTS


def test_it_has_its_own_resource_tag():
    assert resource_tag_for("select_slider", "none") == "Slider"


def test_the_prompt_describes_the_concept_with_several_phrasings():
    """Not one wording - the model has to recognise the idea."""
    assert "select_slider" in SYSTEM_PROMPT
    assert "appealing" in SYSTEM_PROMPT
    assert "agree or" in SYSTEM_PROMPT


@pytest.mark.parametrize("path", ["app/static/app.js", "app/static/quick.js"])
def test_both_review_surfaces_offer_it(path):
    """A card classified this way must not fall back to the first option."""
    source = open(path, encoding="utf-8").read()
    assert '"select_slider"' in source


# -- the generated shape, matched to the real template --------------------


def test_the_generated_xml_matches_the_template():
    xml = generate_question(slider([*SCALE, OptionLine(raw_text="NA")]))

    assert xml.splitlines()[0] == (
        '<select label="A20" randomize="0" ss:questionClassNames="sq-sliderpoints" '
        'uses="sliderpoints.3" values="order">'
    )
    assert "  <comment>${res.Slider}</comment>" in xml
    assert '  <choice label="ch1" value="1">1</choice>' in xml
    assert '  <choice label="ch5" value="5">5</choice>' in xml
    assert '  <choice label="ch99" sliderpoints:OO="1" value="99">NA</choice>' in xml
    assert xml.endswith("</select>")


def test_the_scale_points_are_choices_not_rows():
    xml = generate_question(slider(SCALE))

    assert "<row" not in xml
    assert xml.count("<choice") == 5


def test_a_slider_has_no_validate_element():
    assert "<validate>" not in generate_question(slider(SCALE))


@pytest.mark.parametrize("text", ["NA", "Not applicable", "Don't know", "Prefer not to say"])
def test_any_opt_out_point_is_marked_rather_than_numbered(text):
    xml = generate_question(slider([*SCALE, OptionLine(raw_text=text)]))
    line = next(l for l in xml.splitlines() if text in l)

    assert 'sliderpoints:OO="1"' in line
    assert 'value="6"' not in line, "an opt-out is off the scale, not the next point on it"


def test_an_opt_out_with_a_house_code_keeps_it():
    """"Don't know" is 97 by the Phase 20 table; NA has no code, so it takes 99."""
    labels = {
        line.option.raw_text: line.label
        for line in label_choices([*SCALE, OptionLine(raw_text="Don't know")])
    }
    assert labels["Don't know"] == "ch97"


def test_a_source_code_still_wins_on_a_slider():
    labels = {
        line.option.raw_text: line.label
        for line in label_choices([*SCALE, OptionLine(raw_text="NA", code="98")])
    }
    assert labels["NA"] == "ch98"


def test_scale_points_number_sequentially_without_source_codes():
    options = [OptionLine(raw_text=word) for word in ("Very likely", "Likely", "Unlikely")]
    assert [line.label for line in label_choices(options)] == ["ch1", "ch2", "ch3"]


# -- neighbours it must not become ----------------------------------------


def test_a_radio_grid_is_still_a_radio_grid():
    """False-positive guard: several statements sharing one scale is a grid."""
    xml = generate_question(
        Question(
            label="P1",
            element="radio_grid",
            title=[TextRun(text="For each statement, how strongly do you agree?")],
            rows=[OptionLine(raw_text="Delivers parcels quickly", code="1")],
            cols=[OptionLine(raw_text="Strongly Disagree"), OptionLine(raw_text="Agree")],
            needs_review=False,
        )
    )

    assert "<choice" not in xml
    assert "sliderpoints" not in xml
    assert '<row label="r1">Delivers parcels quickly</row>' in xml


def test_a_plain_radio_is_unaffected():
    xml = generate_question(
        Question(
            label="Q1",
            element="radio",
            title=[TextRun(text="Which brand did you buy?")],
            options=[OptionLine(raw_text="Brand A"), OptionLine(raw_text="Brand B")],
            needs_review=False,
        )
    )

    assert "<choice" not in xml
    assert '<row label="r1" value="1">Brand A</row>' in xml


# -- the dataset entries --------------------------------------------------


def test_a20_is_now_a_scored_entry():
    ids = {example.id for example in load_examples()}
    assert "select_slider_bipolar_scale" in ids
    assert "select_slider_with_opt_out" in ids


def test_the_a20_entry_generates_five_choices():
    example = next(
        e for e in load_examples() if e.id == "select_slider_bipolar_scale"
    )
    xml = generate_question(build_question(example))

    assert xml.count("<choice") == 5
    assert "sliderpoints:OO" not in xml, "a plain scale has no opt-out point"


def test_the_opt_out_entry_generates_its_marked_point():
    example = next(e for e in load_examples() if e.id == "select_slider_with_opt_out")
    xml = generate_question(build_question(example))

    assert '<choice label="ch99" sliderpoints:OO="1" value="99">Not applicable</choice>' in xml
