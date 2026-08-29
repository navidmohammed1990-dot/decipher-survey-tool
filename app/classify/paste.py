"""Splitting pasted questionnaire text into question blocks.

Quick Convert exists because whole-document parsing is where the hard problems
live: cover pages, section furniture, tables that span rows, headers that bleed
between questions. A pasted chunk has none of that — the programmer already did
the segmentation by choosing what to select, exactly as they do in Sublime.

So this stays deliberately small. It normalises what a Word copy-paste produces
and defers to the same label detection the DOCX path uses; it does not
reimplement it.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from app.classify.features import extract_features
from app.classify.lines import SourceLine
from app.models.document import TextRun
from app.parsing.normalize import collapse_whitespace, literal_marker_span
from app.parsing.question_boundaries import BoundaryConfig, match_question_label

#: Word puts a tab between table cells when a row is copied as text. Treating
#: the row as one line is Bug 1's fix in its simplest form: the option's text
#: and its code belong together.
_CELL_SEPARATOR = re.compile(r"[\t]+|(?: {2,}\|)|(?:\|(?= ))")

#: A column-aligned code: two or more spaces, then a short number, at line end.
#: Word pastes table columns as runs of spaces as often as it does tabs, and
#: whitespace collapsing happens before codes are read — so without this the
#: "1" in "Under 18 years  1" is absorbed into the option's own text.
#: Two spaces are required: "I have lived here 20 years" must not lose its 20.
_ALIGNED_CODE = re.compile(r"^(?P<text>\S.*?\S)[ \t]{2,}(?P<code>\d{1,3})\s*$")

#: A label with no question after it still deserves a block.
DEFAULT_LABEL = "Q1"


class PastedQuestion(BaseModel):
    """One question found in a paste."""

    label: str
    lines: list[SourceLine] = Field(default_factory=list)
    raw_label: str | None = None
    synthesised_label: bool = False
    """True when the paste carried no label and one was supplied."""


def join_cells(line: str) -> str:
    """Collapse a copied table row into one line: ``Male\\t1`` -> ``Male | 1``.

    Keeping the separator visible lets the existing feature extractor find the
    trailing code, the same way it does for a real DOCX table row.
    """
    cells = [cell.strip() for cell in _CELL_SEPARATOR.split(line) if cell and cell.strip()]
    if len(cells) > 1:
        return " | ".join(cells)

    aligned = _ALIGNED_CODE.match(line)
    if aligned:
        return f"{aligned.group('text')} | {aligned.group('code')}"
    return line.strip()


def normalise_lines(text: str) -> list[str]:
    """Pasted text as clean, non-empty lines with table rows kept whole."""
    lines: list[str] = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        joined = collapse_whitespace(join_cells(raw))
        if joined:
            lines.append(joined)
    return lines


def _build_line(index: int, text: str) -> SourceLine:
    """One pasted line, with its typed marker stripped and hints observed.

    Mirrors the DOCX path: features are observed on the line as written, while
    the text carried forward has any typed marker removed so that "1." becomes
    the row's code rather than part of its wording.
    """
    span = literal_marker_span(text)
    marker = span[0] if span else None
    body = text[span[1]:].strip() if span else text

    return SourceLine(
        index=index,
        text=body or text,
        runs=[TextRun(text=body or text)],
        kind="paragraph",
        marker=marker,
        literal_marker=marker,
        features=extract_features(text, [TextRun(text=text)]),
    )


def split_questions(
    text: str, config: BoundaryConfig | None = None
) -> tuple[list[PastedQuestion], list[str]]:
    """Split a paste into question blocks, reusing the DOCX label detection.

    A paste with no recognisable label is treated as a single question rather
    than rejected: the programmer selected this chunk deliberately, and one
    unlabelled question is a normal thing to paste.
    """
    lines = normalise_lines(text)
    if not lines:
        return [], ["Nothing to convert - the pasted text is empty."]

    config = config or BoundaryConfig()
    warnings: list[str] = []

    matches = [
        (position, match_question_label(line, config))
        for position, line in enumerate(lines)
    ]
    labelled = [(position, found) for position, found in matches if found]

    if not labelled:
        # Try plain "1." numbering only when nothing else matched, so a numbered
        # option list inside a labelled question is never mistaken for a split.
        labelled = [
            (position, found)
            for position, found in (
                (position, match_question_label(line, config, allow_numeric=True))
                for position, line in enumerate(lines)
            )
            if found
        ]
        if labelled:
            warnings.append(
                f"No Q-style labels found; split on plain numbering into "
                f"{len(labelled)} question(s). Check the split is right."
            )

    if not labelled:
        warnings.append(
            f"No question label found; treated the whole paste as one question "
            f"labelled {DEFAULT_LABEL}."
        )
        return [
            PastedQuestion(
                label=DEFAULT_LABEL,
                synthesised_label=True,
                lines=[_build_line(index, line) for index, line in enumerate(lines)],
            )
        ], warnings

    questions: list[PastedQuestion] = []
    if labelled[0][0] > 0:
        warnings.append(
            f"{labelled[0][0]} line(s) before the first question label were ignored."
        )

    for order, (start, found) in enumerate(labelled):
        label, raw, end = found
        stop = labelled[order + 1][0] if order + 1 < len(labelled) else len(lines)

        # The text after the label on its own line is the question's first line.
        block: list[str] = []
        remainder = lines[start][end:].strip()
        if remainder:
            block.append(remainder)
        block.extend(lines[start + 1:stop])

        questions.append(
            PastedQuestion(
                label=label,
                raw_label=raw,
                lines=[_build_line(index, line) for index, line in enumerate(block)],
            )
        )

    duplicates = {q.label for q in questions if [x.label for x in questions].count(q.label) > 1}
    if duplicates:
        warnings.append(f"Duplicate question label(s): {', '.join(sorted(duplicates))}.")

    return questions, warnings
