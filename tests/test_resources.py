"""Phase 5A — resource tag catalog, subject_type, and SP override."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.classify.classifier import interpret_response
from app.classify.lines import question_lines
from app.generate.resources import (
    RES_PATTERN,
    load_resource_catalog,
    parse_resource_catalog,
    resource_catalog,
    resource_tag_for,
)
from app.generate.xml_generator import generate_question
from app.main import app
from app.models.document import TextRun
from app.models.survey import OptionLine, Question, QuestionDraft
from app.store import draft_store


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def opts(*texts):
    return [OptionLine(raw_text=text) for text in texts]


# -- catalog is parsed, not hardcoded -------------------------------------


def test_catalog_comes_from_the_template_file():
    """Checklist: read from the <res> block at startup, not hardcoded."""
    catalog = resource_catalog()

    assert catalog["SR"] == "Please select one response."
    assert catalog["MRStatement"] == "Please select all that apply for each statement."
    assert "Slider" in catalog, "Phase 18 added the slider tag"
    assert len(catalog) == 13


def test_editing_the_template_changes_the_catalog(tmp_path):
    template = tmp_path / "custom.xml"
    template.write_text('<res label="SR">Totally different wording.</res>')

    assert load_resource_catalog(template) == {"SR": "Totally different wording."}


def test_values_containing_markup_are_extracted_whole():
    """The real template has entries with embedded <br /> and similar."""
    catalog = parse_resource_catalog(
        '<res label="Multi">Line one.<br />Line two.</res>'
    )
    assert catalog["Multi"] == "Line one.<br />Line two."


def test_values_spanning_lines_are_extracted():
    catalog = parse_resource_catalog('<res label="Long">first\n  second</res>')
    assert "first" in catalog["Long"] and "second" in catalog["Long"]


def test_several_entries_on_one_line_do_not_merge():
    """Non-greedy matching: a greedy pattern would swallow both entries."""
    catalog = parse_resource_catalog('<res label="A">one</res><res label="B">two</res>')
    assert catalog == {"A": "one", "B": "two"}


def test_a_full_survey_template_can_be_scanned():
    """The team's real file is a whole survey, not a bare catalog."""
    catalog = parse_resource_catalog(
        '<survey><res label="SR">Pick one.</res>'
        '<radio label="Q1"><title>Hi</title></radio></survey>'
    )
    assert catalog == {"SR": "Pick one."}


def test_a_missing_template_is_not_fatal(tmp_path):
    """Tag selection is pure logic; only the preview text is lost."""
    assert load_resource_catalog(tmp_path / "absent.xml") == {}
    assert resource_tag_for("radio") == "SR"


def test_the_pattern_is_the_one_the_brief_specified():
    assert RES_PATTERN.pattern == r'<res label="([^"]+)">(.*?)</res>'


# -- deterministic selection ----------------------------------------------


@pytest.mark.parametrize(
    "element,subject_type,expected",
    [
        ("radio", "none", "SR"),
        ("checkbox", "none", "MR"),
        ("textarea", "none", "Open"),
        ("text", "none", "Open"),
        ("select", "none", "Ranking"),
        ("number", "none", "Open"),
        ("radio_grid", "brand", "SRBrand"),
        ("radio_grid", "category", "SRCategory"),
        ("radio_grid", "product", "SRProduct"),
        ("radio_grid", "statement", "SRStatement"),
        ("radio_grid", "none", "SRStatement"),
        ("checkbox_grid", "brand", "MRBrand"),
        ("checkbox_grid", "category", "MRCategory"),
        ("checkbox_grid", "product", "MRProduct"),
        ("checkbox_grid", "statement", "MRStatement"),
        ("checkbox_grid", "none", "MRStatement"),
        ("html", "none", None),
    ],
)
def test_the_mapping_table(element, subject_type, expected):
    assert resource_tag_for(element, subject_type) == expected


def test_every_mapped_tag_exists_in_the_catalog():
    catalog = resource_catalog()
    for element in ("radio", "checkbox", "textarea", "text", "select", "number"):
        assert resource_tag_for(element) in catalog
    for element in ("radio_grid", "checkbox_grid"):
        for subject in ("brand", "category", "product", "statement", "none"):
            assert resource_tag_for(element, subject) in catalog


# -- the generator emits references, never resolved text ------------------


def test_generated_comment_is_the_literal_reference():
    """Checklist: ${res.X} syntax, never resolved text."""
    xml_text = generate_question(
        Question(label="Q1", element="radio", title=[TextRun(text="Pick")],
                 options=opts("Yes"), comment_resource="SR")
    )
    assert "<comment>${res.SR}</comment>" in xml_text
    assert "Please select one response." not in xml_text


def test_a_question_with_no_tag_still_gets_the_deterministic_one():
    xml_text = generate_question(
        Question(label="Q1", element="checkbox", title=[TextRun(text="Pick")], options=opts("A"))
    )
    assert "<comment>${res.MR}</comment>" in xml_text


def test_html_takes_no_comment():
    xml_text = generate_question(
        Question(label="Q1", element="html", title=[TextRun(text="Section 2")])
    )
    assert "<comment>" not in xml_text


def test_number_uses_the_open_tag():
    """Phase 6 Bug 4: SR is radio wording and does not fit a numeric field."""
    xml_text = generate_question(Question(label="Q1", element="number",
                                          title=[TextRun(text="Age")]))
    assert "<comment>${res.Open}</comment>" in xml_text
    assert "${res.SR}" not in xml_text


def test_custom_text_overrides_the_tag():
    xml_text = generate_question(
        Question(label="Q1", element="radio", title=[TextRun(text="Pick")], options=opts("A"),
                 comment=[TextRun(text="One-off instruction")], comment_resource=None)
    )
    assert "<comment>One-off instruction</comment>" in xml_text
    assert "${res." not in xml_text


def test_a_chosen_tag_beats_imported_comment_text():
    xml_text = generate_question(
        Question(label="Q1", element="checkbox", title=[TextRun(text="Pick")], options=opts("A"),
                 comment=[TextRun(text="Please select all that apply.")], comment_resource="MR")
    )
    assert "<comment>${res.MR}</comment>" in xml_text


# -- the classifier infers subject_type -----------------------------------


def brand_grid_lines(parsed_sample):
    q6 = next(q for q in parsed_sample.questions if q.label == "Q6")
    return question_lines(parsed_sample, q6)


def test_a_brand_grid_resolves_to_srbrand(parsed_sample):
    """Checklist: brand-named rows get subject_type=brand and SRBrand."""
    outcome = interpret_response(
        {"element": "radio_grid", "title_lines": [0], "row_lines": [4, 5],
         "col_lines": [2, 3], "subject_type": "brand", "confidence": 0.9},
        "Q6", brand_grid_lines(parsed_sample), 0.75,
    )
    assert outcome.question.subject_type == "brand"
    assert outcome.question.comment_resource == "SRBrand"
    assert "<comment>${res.SRBrand}</comment>" in generate_question(outcome.question)


def test_a_checkbox_grid_of_brands_resolves_to_mrbrand(parsed_sample):
    outcome = interpret_response(
        {"element": "checkbox_grid", "title_lines": [0], "row_lines": [4, 5],
         "col_lines": [2, 3], "subject_type": "brand", "confidence": 0.9},
        "Q6", brand_grid_lines(parsed_sample), 0.75,
    )
    assert outcome.question.comment_resource == "MRBrand"


@pytest.mark.parametrize("value", [None, "", "nonsense", 42, "none"])
def test_an_unclear_grid_subject_falls_back_to_statement(parsed_sample, value):
    outcome = interpret_response(
        {"element": "radio_grid", "title_lines": [0], "row_lines": [4, 5],
         "col_lines": [2, 3], "subject_type": value, "confidence": 0.9},
        "Q6", brand_grid_lines(parsed_sample), 0.75,
    )
    assert outcome.question.subject_type == "statement"
    assert outcome.question.comment_resource == "SRStatement"


def test_non_grid_elements_never_carry_a_subject(parsed_sample):
    outcome = interpret_response(
        {"element": "radio", "title_lines": [0], "option_lines": [2],
         "subject_type": "brand", "confidence": 0.9},
        "Q5", brand_grid_lines(parsed_sample), 0.75,
    )
    assert outcome.question.subject_type == "none"
    assert outcome.question.comment_resource == "SR"


def test_the_prompt_asks_for_subject_type():
    from app.classify.classifier import SYSTEM_PROMPT

    assert "subject_type" in SYSTEM_PROMPT
    for subject in ("brand", "category", "product", "statement"):
        assert subject in SYSTEM_PROMPT


def test_the_fallback_still_gets_a_tag(parsed_sample):
    from app.classify.classifier import fallback_question

    question = fallback_question("Q1", brand_grid_lines(parsed_sample))
    assert question.comment_resource == "SR"


# -- SP override over HTTP ------------------------------------------------


@pytest.fixture
def seeded():
    draft_store.replace(QuestionDraft(questions=[
        Question(label="Q1", element="radio", title=[TextRun(text="Pick one")],
                 options=opts("Yes", "No"), comment_resource="SR",
                 confidence=0.9, needs_review=False),
        Question(label="Q6", element="radio_grid", title=[TextRun(text="Rate")],
                 rows=opts("Nike"), cols=opts("Good"), subject_type="brand",
                 comment_resource="SRBrand", confidence=0.9, needs_review=False),
    ]))
    yield
    draft_store.clear()


def test_resources_endpoint_serves_preview_text(client):
    body = client.get("/api/resources").json()

    assert body["available"] is True
    entry = next(r for r in body["resources"] if r["label"] == "SR")
    assert entry["text"] == "Please select one response."


def test_sp_can_switch_to_another_tag(client, seeded):
    body = client.patch("/api/questions/Q1", json={"comment_resource": "Open"}).json()
    assert body["comment_resource"] == "Open"

    xml_text = client.post("/api/generate/Q1").json()["xml"]
    assert "<comment>${res.Open}</comment>" in xml_text


def test_sp_can_switch_to_custom_text(client, seeded):
    """Checklist: any auto-selected tag can be overridden with custom text."""
    body = client.patch(
        "/api/questions/Q1",
        json={"comment_resource": "", "comment": "Answer honestly, please."},
    ).json()
    assert body["comment_resource"] is None

    xml_text = client.post("/api/generate/Q1").json()["xml"]
    assert "<comment>Answer honestly, please.</comment>" in xml_text
    assert "${res." not in xml_text


def test_an_unknown_tag_is_rejected(client, seeded):
    response = client.patch("/api/questions/Q1", json={"comment_resource": "NotATag"})
    assert response.status_code == 422
    assert "not in the resource catalog" in response.text


def test_changing_subject_type_re_derives_the_tag(client, seeded):
    """Q6 is a radio_grid, so product rows must land on SRProduct."""
    body = client.patch("/api/questions/Q6", json={"subject_type": "product"}).json()

    assert body["element"] == "radio_grid"
    assert body["subject_type"] == "product"
    assert body["comment_resource"] == "SRProduct"


def test_changing_a_grid_to_checkbox_re_derives_the_mr_variant(client, seeded):
    body = client.patch("/api/questions/Q6", json={"element": "checkbox_grid"}).json()

    assert body["subject_type"] == "brand"
    assert body["comment_resource"] == "MRBrand"


def test_changing_element_re_derives_the_tag(client, seeded):
    body = client.patch("/api/questions/Q1", json={"element": "checkbox"}).json()
    assert body["comment_resource"] == "MR"


def test_a_custom_comment_survives_an_element_change(client, seeded):
    """A deliberate override must not be silently undone by an unrelated edit."""
    client.patch("/api/questions/Q1", json={"comment_resource": "", "comment": "Mine"})
    body = client.patch("/api/questions/Q1", json={"element": "checkbox"}).json()

    assert body["comment_resource"] is None
    assert body["comment"][0]["text"] == "Mine"


def test_an_invalid_subject_type_is_rejected(client, seeded):
    assert client.patch("/api/questions/Q6", json={"subject_type": "vehicle"}).status_code == 422
