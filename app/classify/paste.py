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
from app.classify.wrapping import merge_wrapped_options
from app.models.document import TextRun
from app.parsing.normalize import collapse_whitespace, literal_marker_span
from app.parsing.question_boundaries import BoundaryConfig, match_question_label

#: Word puts a tab between table cells when a row is copied as text. Treating
#: the row as one line is Bug 1's fix in its simplest form: the option's text
#: and its code belong together.
_CELL_SEPARATOR = re.compile(r"[\t]+|(?: {2,}\|)|(?:\|(?= ))")

#: Three space-aligned columns: text, code, then a directive for that row.
_ALIGNED_ROW_NOTE = re.compile(
    r"^(?P<text>\S.*?\S)[ \t]{2,}(?P<code>\d{1,3})[ \t]{2,}(?P<note>\S.*?)\s*$"
)

#: A column-aligned code: two or more spaces, then a short number, at line end.
#: Word pastes table columns as runs of spaces as often as it does tabs, and
#: whitespace collapsing happens before codes are read — so without this the
#: "1" in "Under 18 years  1" is absorbed into the option's own text.
#: Two spaces are required: "I have lived here 20 years" must not lose its 20.
_ALIGNED_CODE = re.compile(r"^(?P<text>\S.*?\S)[ \t]{2,}(?P<code>\d{1,3})\s*$")

#: A label with no question after it still deserves a block.
DEFAULT_LABEL = "Q1"

#: Strikethrough as it survives a copy-paste into a plain textarea. Word's own
#: formatting is read from the run in the DOCX path; pasted text keeps only
#: this convention.
_STRUCK = re.compile(r"^~~(?P<text>.*?)~~$")


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

    row_note = _ALIGNED_ROW_NOTE.match(line)
    if row_note:
        return f"{row_note.group('text')} | {row_note.group('code')} | {row_note.group('note')}"

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


def _strip_strike(text: str) -> tuple[str, bool]:
    """Unwrap ``~~deleted~~`` on a single line."""
    match = _STRUCK.match(text.strip())
    return (match.group("text").strip(), True) if match else (text, False)


def mark_struck(lines: list[str]) -> list[tuple[str, bool]]:
    """Resolve ``~~`` spans, including ones that run across several lines.

    A struck passage in Word wraps like any other, so the opening and closing
    markers routinely land on different lines once pasted.
    """
    resolved: list[tuple[str, bool]] = []
    open_span = False

    for line in lines:
        stripped = line.strip()
        if not open_span:
            text, struck = _strip_strike(stripped)
            if struck:
                resolved.append((text, True))
                continue
            if stripped.startswith("~~"):
                open_span = True
                resolved.append((stripped[2:].strip(), True))
                continue
            resolved.append((line, False))
        else:
            if stripped.endswith("~~"):
                open_span = False
                resolved.append((stripped[:-2].strip(), True))
            else:
                resolved.append((stripped, True))

    return [(text, struck) for text, struck in resolved if text]


def _build_line(index: int, text: str, struck: bool = False) -> SourceLine:
    """One pasted line, with its typed marker stripped and hints observed.

    Mirrors the DOCX path: features are observed on the line as written, while
    the text carried forward has any typed marker removed so that "1." becomes
    the row's code rather than part of its wording.
    """
    span = literal_marker_span(text)
    marker = span[0] if span else None
    body = text[span[1]:].strip() if span else text

    runs = [TextRun(text=body or text, strike=struck)]
    return SourceLine(
        index=index,
        text=body or text,
        runs=runs,
        kind="paragraph",
        marker=marker,
        literal_marker=marker,
        features=extract_features(text, runs),
    )


#: A scale written across one line needs at least this many points before it
#: reads as a scale rather than an option that happens to contain a pipe.
MIN_SCALE_POINTS = 3


def _is_row_with_note(cells: list[str]) -> bool:
    """Whether these cells read as text | code | directive, not as a scale.

    "17 or younger | 1 | TERMINATE" and "Low | Medium | High" both have three
    cells; only the first has a bare code in the middle.
    """
    return len(cells) == 3 and cells[1].isdigit() and not cells[0].isdigit()


def expand_scale_lines(lines: list[str]) -> list[str]:
    """Break a one-line scale into one line per point.

    A grid's columns often arrive as a single row of pipe-separated labels,
    sometimes followed by their codes on the next row:

        Strongly Disagree | Disagree | Neither | Agree | Strongly Agree
        1                 | 2        | 3       | 4     | 5

    The classifier addresses lines by index and may not rewrite them, so a
    scale packed into one line can never become five columns. Splitting it is
    reflow, the same as rejoining a wrapped option — it decides no roles.
    """
    expanded: list[str] = []
    position = 0

    while position < len(lines):
        cells = [cell.strip() for cell in lines[position].split("|") if cell.strip()]
        if len(cells) < MIN_SCALE_POINTS or _is_row_with_note(cells):
            expanded.append(lines[position])
            position += 1
            continue

        codes: list[str] = []
        if position + 1 < len(lines):
            following = [c.strip() for c in lines[position + 1].split("|") if c.strip()]
            if len(following) == len(cells) and all(c.isdigit() for c in following):
                codes = following

        for index, cell in enumerate(cells):
            expanded.append(f"{cell} | {codes[index]}" if codes else cell)
        position += 2 if codes else 1

    return expanded


def split_questions(
    text: str, config: BoundaryConfig | None = None
) -> tuple[list[PastedQuestion], list[str]]:
    """Split a paste into question blocks, reusing the DOCX label detection.

    A paste with no recognisable label is treated as a single question rather
    than rejected: the programmer selected this chunk deliberately, and one
    unlabelled question is a normal thing to paste.
    """
    lines = expand_scale_lines(normalise_lines(text))
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
                lines=merge_wrapped_options(
                    [
                        _build_line(index, text, struck)
                        for index, (text, struck) in enumerate(mark_struck(lines))
                    ]
                ),
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
                lines=merge_wrapped_options(
                    [
                        _build_line(index, text, struck)
                        for index, (text, struck) in enumerate(mark_struck(block))
                    ]
                ),
            )
        )

    duplicates = {q.label for q in questions if [x.label for x in questions].count(q.label) > 1}
    if duplicates:
        warnings.append(f"Duplicate question label(s): {', '.join(sorted(duplicates))}.")

    return questions, warnings
