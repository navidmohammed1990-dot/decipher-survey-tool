"""Question boundary detection."""

from __future__ import annotations

import docx
import pytest

from app.models.document import ParagraphBlock
from app.parsing.docx_parser import parse_docx
from app.parsing.question_boundaries import (
    BoundaryConfig,
    detect_boundaries,
    normalize_label,
)


def block(index, text, **kwargs):
    return ParagraphBlock(index=index, text=text, **kwargs)


def labels_of(blocks, config=None):
    boundaries, _ = detect_boundaries(blocks, config)
    return [b.label for b in boundaries if not b.is_preamble]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Q5. Which brands?", "Q5"),
        ("Q5) Which brands?", "Q5"),
        ("Q5: Which brands?", "Q5"),
        ("Q5 Which brands?", "Q5"),
        ("Q5 - Which brands?", "Q5"),
        ("[Q7] Bracketed label", "Q7"),
        ("S2. Screener question", "S2"),
        ("QS1. Prefixed screener", "QS1"),
        ("Q10a. A sub-question", "Q10A"),
        ("Q5_1. A sub-question", "Q5_1"),
        ("D3. Demographic", "D3"),
    ],
)
def test_recognises_common_label_styles(text, expected):
    assert labels_of([block(0, text)]) == [expected]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("QD24. Wrapped-option question", "QD24"),
        ("P1. Agreement battery", "P1"),
        ("MP2. Marketplace question", "MP2"),
        ("QZ5. Late-alphabet prefix", "QZ5"),
        ("APP1. Three-letter prefix", "APP1"),
    ],
)
def test_recognises_house_styles_no_whitelist_would_have_covered(text, expected):
    """The point of the shape-based matcher: unseen prefixes still work."""
    assert labels_of([block(0, text)]) == [expected]


@pytest.mark.parametrize(
    "text",
    [
        "Yes 1.",
        "No 2.",
        "Male 1",
        "S2_AGE BANDS",
        "COVID19: impact on your business",
        "Section D. Demographics",
        "Brand A",
        "Please select all that apply.",
        "None of these",
        "Section D. Demographics",
        "Ask Q7 only if Q5 = Brand A.",
        "1997 was a good year",
    ],
)
def test_ignores_text_that_is_not_a_label(text):
    """A shape-based matcher must not swallow ordinary content.

    "Yes 1." and "No 2." are the reason no whitespace is allowed between the
    letters and the digits: permitting it would turn every coded option into a
    question boundary. Missing the rare "Q 12." written with a space is the
    cheaper mistake.

    "S2_AGE BANDS" is why the trailing part is bounded rather than a free
    [A-Za-z0-9_]* — otherwise a derived-variable name reads as a label and
    Phase 11's non-question detection breaks.
    """
    assert labels_of([block(0, text)]) == []


def test_a_question_owns_the_blocks_that_follow_it():
    blocks = [
        block(0, "Q1. First question"),
        block(1, "Option A"),
        block(2, "Option B"),
        block(3, "Q2. Second question"),
        block(4, "Option C"),
    ]
    boundaries, _ = detect_boundaries(blocks)

    assert [b.block_indices for b in boundaries] == [[0, 1, 2], [3, 4]]
    assert [(b.start_index, b.end_index) for b in boundaries] == [(0, 2), (3, 4)]


def test_content_before_the_first_label_becomes_a_preamble():
    blocks = [block(0, "Survey introduction"), block(1, "Q1. First question")]
    boundaries, _ = detect_boundaries(blocks)

    assert boundaries[0].is_preamble
    assert boundaries[0].label is None
    assert boundaries[0].block_indices == [0]


def test_empty_paragraphs_never_start_a_question():
    boundaries, _ = detect_boundaries([block(0, "Q1. Question"), block(1, "")])
    assert len(boundaries) == 1
    assert boundaries[0].block_indices == [0, 1]


def test_numeric_fallback_when_no_prefixed_labels_exist():
    blocks = [block(0, "1. First question"), block(1, "Yes"), block(2, "2. Second question")]
    boundaries, warnings = detect_boundaries(blocks)

    assert [b.label for b in boundaries] == ["1", "2"]
    assert all(b.pattern == "numeric" for b in boundaries)
    assert any("fell back to plain numbering" in w for w in warnings)


def test_numeric_fallback_is_suppressed_when_prefixed_labels_exist():
    """Otherwise every numbered option would look like a new question."""
    blocks = [block(0, "Q1. Real question"), block(1, "1. Brand A"), block(2, "2. Brand B")]
    boundaries, _ = detect_boundaries(blocks)

    assert [b.label for b in boundaries] == ["Q1"]
    assert boundaries[0].block_indices == [0, 1, 2]


def test_numeric_fallback_skips_word_numbered_list_items():
    from app.models.document import ListInfo

    blocks = [
        block(0, "Brand A", list_info=ListInfo(num_id=1, level=0, marker="1.")),
        block(1, "Brand B", list_info=ListInfo(num_id=1, level=0, marker="2.")),
    ]
    boundaries, warnings = detect_boundaries(blocks)

    assert [b.label for b in boundaries if not b.is_preamble] == []
    assert any("No question boundaries detected" in w for w in warnings)


def test_numeric_fallback_can_be_disabled():
    blocks = [block(0, "1. First question")]
    config = BoundaryConfig(allow_numeric_fallback=False)
    assert labels_of(blocks, config) == []


def test_duplicate_labels_are_reported():
    blocks = [block(0, "Q1. First"), block(1, "Q1. Also first")]
    _, warnings = detect_boundaries(blocks)

    assert any("Duplicate question label 'Q1'" in w for w in warnings)


def test_any_prefix_is_accepted_by_default():
    """No enumerated list: the default accepts any label-shaped token."""
    assert labels_of([block(0, "ZZ4. Never-seen prefix")]) == ["ZZ4"]
    assert labels_of([block(0, "XYZ9. Another")]) == ["XYZ9"]


def test_detection_can_still_be_narrowed_deliberately():
    """The allowlist survives as an opt-in for one awkward document."""
    config = BoundaryConfig(prefixes=("ZZ",))
    assert labels_of([block(0, "ZZ4. Custom prefix")], config) == ["ZZ4"]
    assert labels_of([block(0, "Q4. Standard prefix")], config) == []


def test_empty_document_yields_no_boundaries():
    boundaries, warnings = detect_boundaries([])
    assert boundaries == []
    assert warnings


@pytest.mark.parametrize("raw,expected", [("Q 5", "Q5"), ("q5", "Q5"), ("Q5A", "Q5A")])
def test_normalize_label(raw, expected):
    assert normalize_label(raw) == expected


def test_non_breaking_space_does_not_defeat_matching(tmp_path):
    """Word inserts NBSP freely; a label must still be recognised."""
    document = docx.Document()
    document.add_paragraph("Q5. Which brands have you purchased?")
    path = tmp_path / "nbsp.docx"
    document.save(path)

    parsed = parse_docx(path)
    assert [q.label for q in parsed.questions] == ["Q5"]
    assert parsed.questions[0].title_text == "Which brands have you purchased?"
