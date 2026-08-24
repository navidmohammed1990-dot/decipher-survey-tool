"""Flattens a parsed question into the numbered lines the AI points at.

The AI is never given text to rewrite — it is given indices and must answer
with indices. This module owns both ends of that mapping, so the text and
formatting that reach the XML generator are always Phase 1's, never the model's.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.models.document import (
    Block,
    ParagraphBlock,
    ParsedDocument,
    QuestionBoundary,
    TableBlock,
    TextRun,
)
from app.parsing.formatting import merge_runs, runs_to_text, trim_runs_prefix
from app.parsing.normalize import collapse_whitespace, literal_marker_span

LineKind = Literal["paragraph", "table_row", "table_col"]


class SourceLine(BaseModel):
    """One candidate line of a question, as offered to the classifier."""

    index: int
    """Position within this question's line list — the index the AI returns."""
    text: str
    runs: list[TextRun] = Field(default_factory=list)
    kind: LineKind = "paragraph"
    block_index: int | None = None
    marker: str | None = None
    """A list marker stripped from the front of this line, if any."""

    def prompt_form(self) -> str:
        prefix = {"table_row": "[grid row] ", "table_col": "[grid column] "}.get(self.kind, "")
        return f"{self.index}: {prefix}{self.text}"


def _strip_marker(runs: list[TextRun]) -> tuple[list[TextRun], str | None]:
    """Remove a typed list marker ("1. ", "a) ") while preserving formatting.

    The marker becomes the ``r1``/``r2`` label in the generated XML, so leaving
    it in the text would produce ``<row label="r1">1. Price</row>``.
    """
    span = literal_marker_span(runs_to_text(runs))
    if span is None:
        return runs, None
    marker, end = span
    return trim_runs_prefix(runs, end), marker


def question_lines(document: ParsedDocument, question: QuestionBoundary) -> list[SourceLine]:
    """Build the indexed line list for one question.

    The label paragraph contributes the question's *title* runs — Phase 1 has
    already stripped "Q5." from them — so a classifier that picks line 0 as the
    title gets clean text without the label.
    """
    lines: list[SourceLine] = []

    def add(runs: list[TextRun], kind: LineKind, block_index: int | None, marker=None) -> None:
        text = collapse_whitespace(runs_to_text(runs))
        if not text:
            return
        lines.append(
            SourceLine(
                index=len(lines),
                text=text,
                runs=merge_runs(runs),
                kind=kind,
                block_index=block_index,
                marker=marker,
            )
        )

    for block in document.blocks_for(question):
        if isinstance(block, ParagraphBlock):
            if block.index == question.title_block_index:
                add(question.title_runs, "paragraph", block.index)
                continue
            runs, marker = _strip_marker(block.runs)
            add(runs, "paragraph", block.index, marker or (block.list_info.marker if block.list_info else None))
        elif isinstance(block, TableBlock):
            _add_table_lines(block, add)

    return lines


def _add_table_lines(table: TableBlock, add) -> None:
    """Offer a table as grid columns (header row) and grid rows (first column).

    This is the shape a Decipher grid takes: the header row names the scale
    points, the first column names the statements.
    """
    if not table.rows:
        return

    header, *body = table.rows
    has_stub_column = table.n_cols > 1

    for cell in header.cells[1:] if has_stub_column else header.cells:
        add(_cell_runs(cell), "table_col", None)

    for row in body:
        if not row.cells:
            continue
        add(_cell_runs(row.cells[0]), "table_row", None)


def _cell_runs(cell) -> list[TextRun]:
    runs: list[TextRun] = []
    for block in cell.blocks:
        if isinstance(block, ParagraphBlock):
            runs.extend(block.runs)
    return merge_runs(runs)


def lines_by_index(lines: list[SourceLine]) -> dict[int, SourceLine]:
    return {line.index: line for line in lines}


def collect_blocks_text(blocks: list[Block]) -> str:
    """Debug helper: the plain text of a block list."""
    return "\n".join(
        block.text for block in blocks if isinstance(block, ParagraphBlock) and block.text
    )
