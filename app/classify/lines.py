"""Flattens a parsed question into the numbered lines the AI points at.

The AI is never given text to rewrite — it is given indices and must answer
with indices. This module owns both ends of that mapping, so the text and
formatting that reach the XML generator are always Phase 1's, never the model's.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.classify.features import LineFeatures, extract_features
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
    literal_marker: str | None = None
    """The marker only when the author typed it, e.g. the ``97.`` in
    ``97. Other, please specify``.

    Distinct from a marker Word generated: a number the author typed is a code
    they chose, while Word's auto-numbering is only how the list renders.
    """
    features: LineFeatures = Field(default_factory=LineFeatures)
    """Factual observations offered to the classifier as evidence."""

    def prompt_form(self) -> str:
        """The line as the model sees it: raw text plus hints, no role attached."""
        return f'{self.index}: "{self.text}"  [hints: {self.features.as_prompt_hints()}]'


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

    def add(
        runs: list[TextRun],
        kind: LineKind,
        block_index: int | None,
        marker=None,
        source_text: str | None = None,
        literal_marker: str | None = None,
    ) -> None:
        text = collapse_whitespace(runs_to_text(runs))
        if not text:
            return
        # Features are observed on the line as the source wrote it, so a
        # stripped "1." still registers as has_leading_enumeration.
        lines.append(
            SourceLine(
                index=len(lines),
                text=text,
                runs=merge_runs(runs),
                kind=kind,
                block_index=block_index,
                marker=marker,
                literal_marker=literal_marker,
                features=extract_features(
                    source_text if source_text is not None else text,
                    runs,
                    is_table_row=kind in ("table_row", "table_col"),
                ),
            )
        )

    for block in document.blocks_for(question):
        if isinstance(block, ParagraphBlock):
            if block.index == question.title_block_index:
                add(question.title_runs, "paragraph", block.index)
                continue
            original = collapse_whitespace(runs_to_text(block.runs))
            runs, marker = _strip_marker(block.runs)
            add(
                runs,
                "paragraph",
                block.index,
                marker or (block.list_info.marker if block.list_info else None),
                source_text=original,
                literal_marker=marker,
            )
        elif isinstance(block, TableBlock):
            _add_table_lines(block, add)

    from app.classify.wrapping import merge_wrapped_options

    return merge_wrapped_options(lines)


def _add_table_lines(table: TableBlock, add) -> None:
    """Offer a table's rows as lines, one line per row.

    An options table (``Male | 1``) and a grid (statements x scale) are the same
    structure to the parser, and telling them apart is a meaning-level judgment
    the classifier makes — not something to decide here. So every row becomes
    one joined line, and the header cells of a grid-shaped table are offered
    *additionally* as separate lines the classifier may take as columns.

    Emitting cells individually for every table is what dropped the first
    option's text and shifted the rest by one: ``Male | 1`` became a column "1"
    and rows "Female", "Other".
    """
    if not table.rows:
        return

    header_is_separable = _has_sparse_body(table)

    for position, row in enumerate(table.rows):
        cells = [cell for cell in row.cells if cell.text.strip()]
        if not cells:
            continue

        if position == 0 and header_is_separable:
            # Grid shape: the header names the scale points, one per line.
            for cell in cells:
                add(_cell_runs(cell), "table_col", None)
            continue

        add(_joined_row_runs(cells), "table_row", None)


def _has_sparse_body(table: TableBlock) -> bool:
    """True when the table looks like a grid rather than an options list.

    A grid's body rows leave the scale cells empty for the respondent to fill,
    so they carry fewer filled cells than the header. An options table has
    every cell filled on every row. This is a structural observation about the
    table, not a decision about what any line means.
    """
    filled = [
        sum(1 for cell in row.cells if cell.text.strip())
        for row in table.rows
    ]
    if len(filled) < 2 or filled[0] < 2:
        return False
    return any(count < filled[0] for count in filled[1:])


def _joined_row_runs(cells) -> list[TextRun]:
    """One table row as one run list, cells separated by ``|``.

    Keeping the separator visible lets the classifier see that ``Male`` and
    ``1`` came from different cells, and lets the code extractor find the
    trailing value.
    """
    runs: list[TextRun] = []
    for position, cell in enumerate(cells):
        if position:
            runs.append(TextRun(text=" | "))
        runs.extend(_cell_runs(cell))
    return merge_runs(runs)


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


def marker_code(marker: str | None) -> str | None:
    """The numeric value of a typed list marker, e.g. ``97.`` -> ``97``.

    Non-numeric markers (bullets, letters) carry no code.
    """
    if not marker:
        return None
    digits = marker.strip(" .)]}\t")
    return digits if digits.isdigit() else None
