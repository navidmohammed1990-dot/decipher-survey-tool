"""Pydantic models describing a parsed questionnaire document.

These models are the output contract of Phase 1. Everything downstream — the
pre-processor, the local AI, the intermediate survey model — consumes this
shape, so it deliberately stays close to the source document: it records what
the DOCX *says*, not what any question *means*.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TextRun(BaseModel):
    """A stretch of text sharing one set of character formatting.

    Matches the run shape used by the intermediate survey model in the
    workflow document: ``{"text": ..., "bold": ..., "italic": ...}``.
    """

    text: str
    bold: bool = False
    italic: bool = False
    underline: bool = False
    strike: bool = False
    """Struck through in the source: content the author deleted."""
    color: str | None = None
    """Font colour as an uppercase hex string, e.g. ``FF0000``.

    ``None`` means the run uses the document's default colour. Questionnaires
    often mark programmer-only text in a colour, so this is worth preserving
    even though it never reaches the generated XML.
    """

    def formatting_key(self) -> tuple[bool, bool, bool, bool, str | None]:
        return (self.bold, self.italic, self.underline, self.strike, self.color)


class ListInfo(BaseModel):
    """Word numbering information attached to a list paragraph."""

    num_id: int
    level: int
    num_fmt: str | None = None
    """Word's numbering format, e.g. ``decimal``, ``bullet``, ``lowerLetter``."""
    marker: str | None = None
    """Rendered marker text, e.g. ``3.`` or ``b)`` or ``•``.

    Word stores auto-numbering outside the paragraph text, so without this the
    numbering of an option list would be lost entirely.
    """
    from_style: bool = False
    """True when numbering came from the paragraph style rather than the
    paragraph itself."""


class ParagraphBlock(BaseModel):
    kind: Literal["paragraph"] = "paragraph"
    index: int
    """Position of this block in document body order."""
    text: str
    runs: list[TextRun] = Field(default_factory=list)
    style: str | None = None
    heading_level: int | None = None
    alignment: str | None = None
    list_info: ListInfo | None = None
    literal_marker: str | None = None
    """A marker typed literally into the text, e.g. the ``a)`` in ``a) Brand A``.

    Distinct from :attr:`ListInfo.marker`, which Word generated.
    """

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class TableCell(BaseModel):
    row: int
    col: int
    """Column index in the table grid, accounting for horizontal merges."""
    grid_span: int = 1
    v_merge: Literal["restart", "continue"] | None = None
    blocks: list[Block] = Field(default_factory=list)
    text: str = ""


class TableRow(BaseModel):
    index: int
    cells: list[TableCell] = Field(default_factory=list)
    is_header: bool = False


class TableBlock(BaseModel):
    kind: Literal["table"] = "table"
    index: int
    n_rows: int = 0
    n_cols: int = 0
    rows: list[TableRow] = Field(default_factory=list)
    style: str | None = None
    nesting_depth: int = 0


Block = ParagraphBlock | TableBlock


class QuestionBoundary(BaseModel):
    """One detected question region of the document.

    Holds indices into :attr:`ParsedDocument.blocks` rather than copies of the
    blocks, so a document is never serialised twice.
    """

    label: str | None
    """Normalised label, e.g. ``Q5``. ``None`` for the preamble segment."""
    raw_label: str | None = None
    """The label exactly as it appeared in the source, e.g. ``Q 5.``"""
    block_indices: list[int] = Field(default_factory=list)
    start_index: int
    end_index: int
    """Inclusive index of the last block belonging to this question."""
    title_block_index: int | None = None
    title_runs: list[TextRun] = Field(default_factory=list)
    """Runs of the label paragraph with the label prefix removed, formatting
    preserved."""
    title_text: str = ""
    pattern: str | None = None
    """Which boundary pattern matched, for debugging detection behaviour."""
    is_preamble: bool = False


class DocumentStats(BaseModel):
    paragraphs: int = 0
    non_empty_paragraphs: int = 0
    tables: int = 0
    runs: int = 0
    bold_runs: int = 0
    italic_runs: int = 0
    questions: int = 0


class ParsedDocument(BaseModel):
    source_filename: str | None = None
    blocks: list[Block] = Field(default_factory=list)
    questions: list[QuestionBoundary] = Field(default_factory=list)
    stats: DocumentStats = Field(default_factory=DocumentStats)
    warnings: list[str] = Field(default_factory=list)

    def iter_all_blocks(self):
        """Yield every block, including those nested inside table cells."""

        def walk(blocks):
            for block in blocks:
                yield block
                if isinstance(block, TableBlock):
                    for row in block.rows:
                        for cell in row.cells:
                            yield from walk(cell.blocks)

        yield from walk(self.blocks)

    def block(self, index: int) -> Block | None:
        """Look up a block by its ``index``.

        Indices are unique across the whole document, including nested blocks,
        so they cannot be used as positions into :attr:`blocks`.
        """
        for block in self.iter_all_blocks():
            if block.index == index:
                return block
        return None

    def blocks_for(self, question: QuestionBoundary) -> list[Block]:
        wanted = set(question.block_indices)
        return [block for block in self.blocks if block.index in wanted]


TableCell.model_rebuild()
