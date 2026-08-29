"""Phase 3 — deterministic XML generation.

These tests pin the canonical template's attribute sets. If one fails after a
change to `xml_generator.ELEMENT_SPECS`, the spec table is wrong, not the test.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import pytest

from app.generate.export import check_well_formed, wrap_survey
from app.generate.labels import label_cols, label_rows
from app.generate.text import clean, escape_xml, option_markup, runs_to_markup, to_ascii
from app.generate.xml_generator import (
    ELEMENT_SPECS,
    UnsupportedElementError,
    generate_fragment,
    generate_question,
    generate_questions,
)
from app.models.document import TextRun
from app.models.survey import SUPPORTED_ELEMENTS, OptionLine, Question


def opts(*texts):
    return [OptionLine(raw_text=text) for text in texts]


def runs(text):
    return [TextRun(text=text)]


def q(label="Q1", element="radio", **kwargs):
    kwargs.setdefault("title", runs("A question"))
    return Question(label=label, element=element, **kwargs)


def attrs_of(xml_text):
    """Parse the opening tag's attributes, preserving order."""
    opening = xml_text.split("\n", 1)[0]
    return re.findall(r'([\w:]+)="([^"]*)"', opening)


# -- text cleanup ---------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("“Smart”", '"Smart"'),
        ("don’t", "don't"),
        ("a — b", "a - b"),
        ("a – b", "a - b"),
        ("wait…", "wait..."),
        ("nb space", "nb space"),
    ],
)
def test_typographic_characters_become_ascii(raw, expected):
    assert to_ascii(raw) == expected


def test_bare_ampersand_is_escaped():
    assert escape_xml("Tom & Jerry") == "Tom &amp; Jerry"


@pytest.mark.parametrize("entity", ["&amp;", "&lt;", "&#233;", "&#x1F600;", "&quot;"])
def test_existing_entities_are_not_double_escaped(entity):
    assert escape_xml(f"a {entity} b") == f"a {entity} b"


def test_escaping_is_idempotent():
    once = escape_xml("Q&A <tag>")
    assert escape_xml(once) == once


def test_angle_brackets_in_source_text_cannot_become_tags():
    assert "<script>" not in clean("<script>alert(1)</script>")
    assert "&lt;script&gt;" in clean("<script>alert(1)</script>")


def test_runs_become_bold_and_italic_markup():
    markup = runs_to_markup([
        TextRun(text="plain "), TextRun(text="bold", bold=True),
        TextRun(text=" and "), TextRun(text="italic", italic=True),
    ])
    assert markup == "plain <b>bold</b> and <i>italic</i>"


def test_a_run_that_is_both_bold_and_italic_nests():
    markup = runs_to_markup([TextRun(text="both", bold=True, italic=True)])
    assert markup == "<i><b>both</b></i>"


def test_option_markup_applies_line_level_formatting():
    assert option_markup("Brand A", bold=True) == "<b>Brand A</b>"
    assert option_markup("Brand A") == "Brand A"


# -- labelling ------------------------------------------------------------


def test_rows_number_sequentially():
    assert [l.label for l in label_rows(opts("A", "B", "C"), element="radio")] == [
        "r1", "r2", "r3",
    ]


@pytest.mark.parametrize(
    "text",
    ["Other (please specify)", "Other, please specify", "OTHER - SPECIFY", "other: specify below"],
)
def test_other_specify_becomes_r91_with_an_open_box(text):
    (line,) = label_rows(opts(text), element="radio")
    assert line.label == "r91"
    assert line.attrs["open"] == "1"
    assert line.attrs["openSize"] == "25"
    assert line.attrs["randomize"] == "0"


@pytest.mark.parametrize("text", ["Other", "Please specify", "Another brand"])
def test_other_without_specify_is_an_ordinary_row(text):
    (line,) = label_rows(opts(text), element="radio")
    assert line.label == "r1"
    assert line.attrs == {}


@pytest.mark.parametrize("text", ["None of the above", "None of these", "NONE OF THESE"])
def test_none_of_the_above_becomes_r99(text):
    (line,) = label_rows(opts(text), element="radio")
    assert line.label == "r99"
    assert line.attrs["randomize"] == "0"


def test_exclusive_is_added_for_checkbox_only():
    """Radio needs no exclusive — only one answer is possible anyway."""
    (checkbox,) = label_rows(opts("None of these"), element="checkbox")
    (radio,) = label_rows(opts("None of these"), element="radio")

    assert checkbox.attrs["exclusive"] == "1"
    assert "exclusive" not in radio.attrs


def test_the_sequential_counter_skips_special_rows():
    labels = [l.label for l in label_rows(
        opts("A", "Other, please specify", "B", "None of these", "C"), element="radio"
    )]
    assert labels == ["r1", "r91", "r2", "r99", "r3"]


def test_columns_number_sequentially_without_the_r91_r99_convention():
    labels = [l.label for l in label_cols(opts("A", "None of these", "B"))]
    assert labels == ["c1", "c2", "c3"]


def test_columns_still_take_open_handling_for_other_specify():
    lines = label_cols(opts("A", "Other, please specify"))
    assert lines[1].label == "c2"
    assert lines[1].attrs["open"] == "1"


# -- element shapes -------------------------------------------------------


def test_every_element_either_has_a_spec_or_renders_nothing():
    """Three outcomes deliberately have no XML shape.

    not_a_question is programmer content, custom_complex needs bespoke
    scripting, and excluded was struck through in the source. Each renders
    nothing rather than an approximation.
    """
    from app.models.survey import NO_XML_ELEMENTS

    assert set(ELEMENT_SPECS) | NO_XML_ELEMENTS == set(SUPPORTED_ELEMENTS)
    assert set(ELEMENT_SPECS).isdisjoint(NO_XML_ELEMENTS)


def test_radio_matches_the_canonical_attribute_set():
    xml_text = generate_question(q(element="radio", options=opts("Yes", "No")))

    assert attrs_of(xml_text) == [
        ("label", "Q1"),
        ("atm1d:showInput", "0"),
        ("atm1d:viewMode", "vertical"),
        ("randomize", "0"),
        ("ss:listDisplay", "1"),
        ("uses", "atm1d.10"),
        ("values", "order"),
    ]
    assert "<comment>${res.SR}</comment>" in xml_text
    assert "<validate>CheckBlank(1,Q1)</validate>" in xml_text


def test_checkbox_matches_the_canonical_attribute_set():
    xml_text = generate_question(q(element="checkbox", options=opts("Yes", "No")))

    assert attrs_of(xml_text) == [
        ("label", "Q1"),
        ("atleast", "1"),
        ("atm1d:showInput", "0"),
        ("atm1d:viewMode", "vertical"),
        ("fwidth", "1000"),
        ("randomize", "0"),
        ("ss:listDisplay", "1"),
        ("uses", "atm1d.10"),
    ]
    assert "<comment>${res.MR}</comment>" in xml_text
    assert "<validate>CheckBlank(1,Q1)</validate>" in xml_text


def test_radio_rows_carry_value_matching_the_label_suffix():
    xml_text = generate_question(q(element="radio", options=opts("Yes", "No", "None of these")))

    assert '<row label="r1" value="1">Yes</row>' in xml_text
    assert '<row label="r2" value="2">No</row>' in xml_text
    assert '<row label="r99" value="99" randomize="0">None of these</row>' in xml_text


def test_checkbox_rows_omit_value():
    xml_text = generate_question(q(element="checkbox", options=opts("Yes", "No")))

    assert '<row label="r1">Yes</row>' in xml_text
    assert "value=" not in xml_text


@pytest.mark.parametrize(
    "element,expected_attrs,comment,validate",
    [
        ("textarea", [("height", "10"), ("optional", "0"), ("randomize", "0"), ("width", "50")],
         "${res.Open}", "CheckBlank(2,Q1)"),
        ("text", [("optional", "0"), ("randomize", "0"), ("size", "25")],
         "${res.Open}", "CheckBlank(2,Q1)"),
        ("number", [("size", "3"), ("optional", "0")], "${res.Open}", None),
        ("select", [("optional", "0")], "${res.Ranking}", None),
    ],
)
def test_open_and_simple_element_shapes(element, expected_attrs, comment, validate):
    xml_text = generate_question(q(element=element, options=opts("A", "B")))

    assert attrs_of(xml_text) == [("label", "Q1"), *expected_attrs]
    if comment:
        assert f"<comment>{comment}</comment>" in xml_text
    else:
        assert "<comment>" not in xml_text
    if validate:
        assert f"<validate>{validate}</validate>" in xml_text
    else:
        assert "<validate>" not in xml_text


def test_select_rows_are_plain_and_sequential():
    xml_text = generate_question(q(element="select", options=opts("A", "B", "C")))

    assert '<row label="r1">A</row>' in xml_text
    assert '<row label="r3">C</row>' in xml_text
    assert "value=" not in xml_text


def test_radio_grid_has_rows_and_cols_without_values():
    xml_text = generate_question(
        q(element="radio_grid", rows=opts("Statement 1"), cols=opts("Agree", "Disagree"))
    )

    assert attrs_of(xml_text) == [("label", "Q1"), ("randomize", "0")]
    assert xml_text.startswith("<radio ")
    assert '<row label="r1">Statement 1</row>' in xml_text
    assert '<col label="c1">Agree</col>' in xml_text
    assert "value=" not in xml_text
    assert "<comment>${res.SRStatement}</comment>" in xml_text


def test_checkbox_grid_adds_atleast():
    xml_text = generate_question(
        q(element="checkbox_grid", rows=opts("Statement 1"), cols=opts("Yes"))
    )

    assert attrs_of(xml_text) == [("label", "Q1"), ("atleast", "1"), ("randomize", "0")]
    assert xml_text.startswith("<checkbox ")
    assert "<comment>${res.MRStatement}</comment>" in xml_text


def test_grids_read_rows_not_options():
    """A grid's answer list lives in rows/cols; options must be ignored."""
    xml_text = generate_question(
        q(element="radio_grid", rows=opts("Real row"), cols=opts("C"), options=opts("Ignored"))
    )
    assert "Real row" in xml_text
    assert "Ignored" not in xml_text


def test_html_is_minimal():
    xml_text = generate_question(q(element="html", title=runs("Section 2 begins here")))
    assert xml_text == '<html label="Q1" where="survey">Section 2 begins here</html>'


def test_a_supplied_comment_overrides_the_default():
    xml_text = generate_question(
        q(element="radio", comment=runs("Custom instruction"), options=opts("A"))
    )
    assert "<comment>Custom instruction</comment>" in xml_text
    assert "${res.SR}" not in xml_text


def test_formatting_survives_into_the_generated_xml():
    xml_text = generate_question(q(
        element="checkbox",
        title=[TextRun(text="Which "), TextRun(text="brands", bold=True)],
        options=[OptionLine(raw_text="Brand A", italic=True)],
    ))
    assert "<title>Which <b>brands</b></title>" in xml_text
    assert '<row label="r1"><i>Brand A</i></row>' in xml_text


def test_dev_notes_never_reach_the_xml():
    xml_text = generate_question(
        q(element="radio", options=opts("A"), dev_notes="SP: check the routing here")
    )
    assert "check the routing" not in xml_text


def test_unsupported_element_is_rejected():
    with pytest.raises(UnsupportedElementError):
        generate_question(Question(label="Q1", element="dropdown"))


# -- fragments and assembly ----------------------------------------------


def test_every_fragment_ends_with_a_blank_line_then_suspend():
    fragment = generate_fragment(q(element="radio", options=opts("A")))
    assert fragment.endswith("</radio>\n\n<suspend/>")


def test_assembly_emits_one_suspend_per_question():
    xml_text = generate_questions([
        q(label="Q1", element="radio", options=opts("A")),
        q(label="Q2", element="text"),
    ])
    assert xml_text.count("<suspend/>") == 2


# -- the guarantees -------------------------------------------------------


def test_output_is_byte_for_byte_identical_across_runs():
    question = q(element="checkbox", options=opts("A", "Other, please specify", "None of these"))
    outputs = {generate_fragment(question.model_copy(deep=True)) for _ in range(25)}
    assert len(outputs) == 1


def test_output_contains_no_timestamp_or_random_token():
    xml_text = generate_questions([q(element="radio", options=opts("A"))])
    assert not re.search(r"\d{4}-\d{2}-\d{2}", xml_text)
    assert "uuid" not in xml_text.lower()


@pytest.mark.parametrize("element", SUPPORTED_ELEMENTS)
def test_every_element_produces_well_formed_xml(element):
    question = q(element=element, options=opts("A", "B"), rows=opts("R"), cols=opts("C"))
    result = check_well_formed(generate_fragment(question))
    assert result.ok, result.error


def test_namespaces_are_declared_at_the_root_not_per_fragment():
    xml_text = generate_questions([q(element="radio", options=opts("A"))])

    assert "xmlns:" not in xml_text, "fragments must not declare namespaces"
    assert "xmlns:atm1d" in wrap_survey(xml_text)
    assert "xmlns:ss" in wrap_survey(xml_text)


def test_wrapped_export_parses_and_keeps_prefixed_attributes():
    xml_text = generate_questions([q(element="radio", options=opts("A"))])
    root = ET.fromstring(wrap_survey(xml_text))

    radio = root.find("radio")
    assert radio is not None
    assert radio.get("label") == "Q1"
    # The prefixed attribute survived namespace resolution.
    assert any("showInput" in key for key in radio.attrib)


def test_ampersands_from_the_questionnaire_survive_a_parse():
    xml_text = generate_questions([q(element="radio", title=runs("Tom & Jerry"), options=opts("A"))])
    root = ET.fromstring(wrap_survey(xml_text))

    assert root.find("radio/title").text == "Tom & Jerry"


def test_malformed_input_is_reported_not_raised():
    result = check_well_formed("<radio label='Q1'><title>unclosed</radio>")
    assert result.ok is False
    assert result.error
