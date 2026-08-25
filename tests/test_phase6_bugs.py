"""Phase 6 — the bugs found by comparing generated XML against the source DOCX.

Each test names the bug it pins. The Tasmania questionnaire is rebuilt in the
shape that produced the original report.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.classify.lines import question_lines
from app.generate.labels import label_cols, label_rows
from app.generate.xml_generator import generate_question
from app.main import app
from app.models.document import TextRun
from app.models.survey import OptionLine, Question
from app.parsing.docx_parser import parse_docx
from app.store import draft_store
from tests.fixtures.build_tasmania import build_tasmania


@pytest.fixture(scope="module")
def tasmania(tmp_path_factory):
    return parse_docx(build_tasmania(tmp_path_factory.mktemp("tas") / "tasmania.docx"))


def lines_for(parsed, label):
    boundary = next(q for q in parsed.questions if q.label == label)
    return question_lines(parsed, boundary)


def opts(*texts):
    return [OptionLine.from_text(text) for text in texts]


# -- Bug 1: table option text and its code land on separate lines ----------


def test_a_table_row_becomes_one_complete_line(tasmania):
    """The root cause: cells were emitted individually, losing the first
    option's text and shifting everything by one."""
    texts = [line.text for line in lines_for(tasmania, "Q1.2")]

    assert "Male | 1" in texts
    assert "Female | 2" in texts
    assert "Other | 97" in texts


def test_no_option_text_is_dropped(tasmania):
    """Q1.6 must keep all four options, none lost to cell splitting."""
    texts = [line.text for line in lines_for(tasmania, "Q1.6")]

    assert "Car | 1" in texts
    assert "Motorcycle | 2" in texts
    assert "Truck | 3" in texts
    assert "None of these | 99" in texts


def test_a_grid_still_separates_rows_from_columns(sample_docx):
    """Joining rows must not break grids, whose columns live in the header."""
    parsed = parse_docx(sample_docx)
    lines = lines_for(parsed, "Q6")

    assert [l.text for l in lines if l.kind == "table_col"] == [
        "Statement", "Agree", "Disagree",
    ]
    assert [l.text for l in lines if l.kind == "table_row"] == [
        "The brand is good value", "The brand is easy to find",
    ]


# -- Bug 2 (boundary half): the next question's header bleeding backwards ---


def test_a_type_header_belongs_to_the_question_it_introduces(tasmania):
    """"ASK ALL, SC" sits above Q1.2's label, so plain segmentation left it at
    the tail of Q1.1 — where it was eligible to become an option."""
    assert lines_for(tasmania, "Q1.2")[0].text == "ASK ALL, SC"
    assert lines_for(tasmania, "Q1.6")[0].text == "ASK ALL, MC"

    assert "ASK ALL, SC" not in [line.text for line in lines_for(tasmania, "Q1.1")]


def test_post_question_routing_stays_with_its_own_question(tasmania):
    """Only headers move. A trailing QUALIFY IF belongs where it was written."""
    texts = [line.text for line in lines_for(tasmania, "Q1.6")]
    assert "QUALIFY IF Q1.4 = 1-2 AND Q1.6 <> 99" in texts


# -- Bug 3: source-provided codes are preserved, not renumbered ------------


def test_source_codes_become_the_row_value(tasmania):
    """Other was coded 97; sequential renumbering to 3 broke the data tables."""
    question = Question(
        label="Q1.2", element="radio", title=[TextRun(text="Gender?")],
        options=opts("Male | 1", "Female | 2", "Other | 97"),
    )
    xml_text = generate_question(question)

    assert '<row label="r1" value="1">Male</row>' in xml_text
    assert '<row label="r2" value="2">Female</row>' in xml_text
    assert '<row label="r97" value="97">Other</row>' in xml_text


def test_the_code_is_not_left_in_the_option_text():
    (option,) = opts("Male | 1")
    assert option.raw_text == "Male"
    assert option.code == "1"


def test_explicit_codes_take_priority_over_the_r91_convention():
    labelled = label_rows(opts("Other, please specify | 5"), element="radio")

    assert labelled[0].label == "r5", "an explicit code outranks the convention"
    assert labelled[0].attrs["open"] == "1", "but the text box still follows the text"


def test_the_convention_remains_the_fallback_when_no_code_is_given():
    labels = [l.label for l in label_rows(
        opts("Brand A", "Other, please specify", "None of these"), element="checkbox"
    )]
    assert labels == ["r1", "r91", "r99"]


def test_sequential_numbering_is_the_last_resort():
    labels = [l.label for l in label_rows(opts("Yes", "No"), element="radio")]
    assert labels == ["r1", "r2"]


def test_mixed_coded_and_uncoded_options_do_not_collide():
    labels = [l.label for l in label_rows(opts("Yes", "Other | 97", "No"), element="radio")]
    assert labels == ["r1", "r97", "r2"]


def test_columns_honour_codes_too():
    assert [c.label for c in label_cols(opts("Agree | 1", "Disagree | 2"))] == ["c1", "c2"]


def test_a_non_numeric_code_falls_through_safely():
    labels = [l.label for l in label_rows([OptionLine(raw_text="Yes", code="n/a")], element="radio")]
    assert labels == ["r1"]


# -- Bug 4: number had radio wording -------------------------------------


def test_number_uses_the_open_resource_tag():
    xml_text = generate_question(
        Question(label="Q1.1", element="number", title=[TextRun(text="What is your age?")])
    )
    assert "<comment>${res.Open}</comment>" in xml_text
    assert "${res.SR}" not in xml_text


# -- the combined check from the brief ------------------------------------


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_routing_text_never_reaches_the_generated_xml(client):
    """Routing and quota logic is the programmer's job, not the tool's."""
    draft_store.clear()
    question = Question(
        label="Q1.2", element="radio", title=[TextRun(text="Gender?")],
        options=opts("Male | 1", "Female | 2", "Other | 97"),
        routing_notes=["RANDOMLY ASSIGN OTHER INTO MALE/FEMALE QUOTAS", "ASK ALL, SC"],
        needs_review=False,
    )
    xml_text = client.post(
        "/api/generate", json={"questions": [question.model_dump(mode="json")]}
    ).json()["xml"]

    assert "RANDOMLY ASSIGN" not in xml_text
    assert "ASK ALL" not in xml_text
    assert xml_text.count("<row") == 3
    draft_store.clear()
