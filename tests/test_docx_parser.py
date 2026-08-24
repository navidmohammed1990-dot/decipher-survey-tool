"""End-to-end parsing of a questionnaire document."""

from __future__ import annotations

import docx
import pytest

from app.models.document import ParagraphBlock, TableBlock
from app.parsing.docx_parser import DocxParseError, parse_docx


def paragraphs(parsed):
    return [b for b in parsed.blocks if isinstance(b, ParagraphBlock)]


def tables(parsed):
    return [b for b in parsed.blocks if isinstance(b, TableBlock)]


def test_parses_without_warnings(parsed_sample):
    assert parsed_sample.warnings == []
    assert parsed_sample.source_filename.endswith(".docx")


def test_finds_every_question(parsed_sample):
    labels = [q.label for q in parsed_sample.questions if not q.is_preamble]
    assert labels == ["S1", "Q5", "Q6", "Q7"]
    assert parsed_sample.stats.questions == 4


def test_content_before_the_first_question_becomes_a_preamble(parsed_sample):
    preamble = parsed_sample.questions[0]
    assert preamble.is_preamble
    assert preamble.label is None
    assert "Brand Tracker 2026" in parsed_sample.block(preamble.start_index).text


def test_question_title_excludes_the_label(parsed_sample):
    q5 = next(q for q in parsed_sample.questions if q.label == "Q5")
    assert q5.title_text.startswith("Which of the following brands")
    assert "Q5" not in q5.title_text


def test_title_formatting_survives_label_removal(parsed_sample):
    """The workflow doc makes the parser the source of truth for formatting."""
    q5 = next(q for q in parsed_sample.questions if q.label == "Q5")

    bold = [r.text for r in q5.title_runs if r.bold]
    italic = [r.text for r in q5.title_runs if r.italic]

    assert bold == ["purchased"]
    assert italic == ["6 months"]


def test_bold_instruction_paragraph_is_preserved(parsed_sample):
    q5 = next(q for q in parsed_sample.questions if q.label == "Q5")
    instruction = next(
        b for b in parsed_sample.blocks_for(q5)
        if isinstance(b, ParagraphBlock) and "select all that apply" in b.text.lower()
    )
    assert all(run.bold for run in instruction.runs)


def test_options_carry_word_numbering_markers(parsed_sample):
    q5 = next(q for q in parsed_sample.questions if q.label == "Q5")
    options = [
        b for b in parsed_sample.blocks_for(q5)
        if isinstance(b, ParagraphBlock) and b.list_info is not None
    ]

    assert [b.text for b in options] == ["Brand A", "Brand B", "Brand C", "None of these"]
    # The marker text is not in the paragraph text at all; it comes from
    # numbering.xml, and would be lost without the numbering resolver.
    assert all(b.list_info.marker for b in options)
    assert all(b.list_info.num_fmt == "decimal" for b in options)


def test_literal_markers_typed_into_text_are_detected(parsed_sample):
    q7 = next(q for q in parsed_sample.questions if q.label == "Q7")
    markers = [
        b.literal_marker for b in parsed_sample.blocks_for(q7)
        if isinstance(b, ParagraphBlock) and b.literal_marker
    ]
    assert markers == ["1.", "2.", "3."]


def test_table_stays_attached_to_the_question_above_it(parsed_sample):
    """python-docx's separate paragraphs/tables lists would lose this."""
    q6 = next(q for q in parsed_sample.questions if q.label == "Q6")
    blocks = parsed_sample.blocks_for(q6)

    table = next(b for b in blocks if isinstance(b, TableBlock))
    stem = blocks[0]

    assert stem.index < table.index
    assert table.n_rows == 3 and table.n_cols == 3


def test_table_cell_content_is_parsed_with_formatting(parsed_sample):
    table = tables(parsed_sample)[0]
    header_cells = table.rows[0].cells

    assert [c.text for c in header_cells] == ["Statement", "Agree", "Disagree"]
    assert all(cell.blocks[0].runs[0].bold for cell in header_cells)
    assert table.rows[1].cells[0].text == "The brand is good value"


def test_blocks_appear_in_document_order(parsed_sample):
    indices = [b.index for b in parsed_sample.blocks]
    assert indices == sorted(indices)


def test_block_indices_are_unique_across_nested_content(parsed_sample):
    indices = [b.index for b in parsed_sample.iter_all_blocks()]
    assert len(indices) == len(set(indices))


def test_nested_blocks_sort_after_their_parent_table(parsed_sample):
    table = tables(parsed_sample)[0]
    nested = [b.index for row in table.rows for c in row.cells for b in c.blocks]
    assert all(index > table.index for index in nested)


def test_stats_are_counted(parsed_sample):
    stats = parsed_sample.stats
    assert stats.tables == 1
    assert stats.bold_runs >= 5
    assert stats.italic_runs >= 2
    assert stats.non_empty_paragraphs > 0


def test_heading_and_alignment_metadata(parsed_sample):
    title = paragraphs(parsed_sample)[0]
    assert title.style == "Title"
    assert title.heading_level == 0

    closing = paragraphs(parsed_sample)[-1]
    assert closing.alignment == "CENTER"


def test_parsed_document_is_json_serialisable(parsed_sample):
    payload = parsed_sample.model_dump_json()
    assert '"kind":"table"' in payload
    assert '"bold":true' in payload


def test_rejects_a_non_docx_file(tmp_path):
    bogus = tmp_path / "questionnaire.docx"
    bogus.write_bytes(b"this is not a zip archive")

    with pytest.raises(DocxParseError):
        parse_docx(bogus)


def test_rejects_a_missing_file(tmp_path):
    with pytest.raises(DocxParseError):
        parse_docx(tmp_path / "absent.docx")


def test_empty_document_parses_and_warns(tmp_path):
    path = tmp_path / "empty.docx"
    docx.Document().save(path)

    parsed = parse_docx(path)
    assert parsed.stats.questions == 0
    assert any("No question boundaries" in w for w in parsed.warnings)


def test_parses_from_a_binary_stream(sample_docx):
    with open(sample_docx, "rb") as stream:
        parsed = parse_docx(stream, filename="uploaded.docx")

    assert parsed.source_filename == "uploaded.docx"
    assert parsed.stats.questions == 4
