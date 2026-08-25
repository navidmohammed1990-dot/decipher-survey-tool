"""Detection of question boundaries in a parsed document.

Boundary detection answers "where does each question start and stop", not
"what kind of question is this". Classification is the local AI's job in a
later phase; keeping the two apart is what lets the parser stay deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.models.document import Block, ParagraphBlock, QuestionBoundary, TextRun
from app.parsing.formatting import runs_to_text, trim_runs_prefix
from app.parsing.normalize import collapse_whitespace, normalize_for_matching

#: Label prefixes seen on real questionnaires. Longest first so that "QS1"
#: does not match as "Q" followed by stray text.
DEFAULT_PREFIXES = (
    "INTRO", "SCREEN", "DEM", "QS", "SC", "HD", "QA", "QB", "QC",
    "Q", "S", "A", "B", "C", "D", "F",
)


@dataclass
class BoundaryConfig:
    """Tuning knobs for boundary detection."""

    prefixes: tuple[str, ...] = DEFAULT_PREFIXES
    allow_numeric_fallback: bool = True
    """When no prefixed label exists anywhere, treat "1." / "2)" paragraphs as
    question starts."""
    max_label_digits: int = 3
    extra_patterns: tuple[str, ...] = field(default_factory=tuple)


def _build_prefixed_pattern(config: BoundaryConfig) -> re.Pattern:
    prefixes = sorted(config.prefixes, key=len, reverse=True)
    alternation = "|".join(re.escape(p) for p in prefixes)
    return re.compile(
        rf"""^[ \t]*
        \[?[ \t]*
        (?P<label>
            (?:{alternation})
            [ \t]*\d{{1,{config.max_label_digits}}}
            [A-Za-z]?
            (?:[._-]\d{{1,{config.max_label_digits}}})?
        )
        [ \t]*\]?
        (?P<sep>[.):\-–—]+[ \t]*|[ \t]+|$)
        """,
        re.VERBOSE,
    )


#: Fallback for questionnaires that number questions without a letter prefix.
NUMERIC_PATTERN = re.compile(r"^[ \t]*(?P<label>\d{1,3})(?P<sep>[.)][ \t]*)")

#: A house-style header introducing the question that follows it, e.g.
#: "ASK ALL, SC" or "[MC]". These sit *above* their question's label, so they
#: land at the tail of the previous question unless reassigned.
QUESTION_HEADER = re.compile(
    r"""^[ \t]*[\[\(]?[ \t]*
    (?:
        ASK[ \t]+(?:ALL|IF|ONLY[ \t]+IF)\b
      | (?:SC|MC|OE|NUM|GRID)[ \t]*[\],;:.\)]?[ \t]*$
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)


def normalize_label(raw: str) -> str:
    return re.sub(r"[ \t]+", "", raw).upper()


@dataclass
class _Match:
    block_index: int
    label: str
    raw_label: str
    end_offset: int
    pattern: str


def _find_matches(
    blocks: list[Block], pattern: re.Pattern, name: str, *, skip_list_items: bool
) -> list[_Match]:
    matches: list[_Match] = []
    for block in blocks:
        if not isinstance(block, ParagraphBlock) or block.is_empty:
            continue
        if skip_list_items and block.list_info is not None:
            # An auto-numbered paragraph is far more likely an option than a
            # question start.
            continue
        match = pattern.match(normalize_for_matching(block.text))
        if match is None:
            continue
        matches.append(
            _Match(
                block_index=block.index,
                label=normalize_label(match.group("label")),
                raw_label=match.group(0).strip(),
                end_offset=match.end(),
                pattern=name,
            )
        )
    return matches


def detect_boundaries(
    blocks: list[Block], config: BoundaryConfig | None = None
) -> tuple[list[QuestionBoundary], list[str]]:
    """Split ``blocks`` into question segments.

    Returns the segments and any warnings raised while detecting them. Only
    top-level blocks are considered: a table belongs to the question it follows,
    and paragraphs inside its cells are grid rows, not new questions.
    """
    config = config or BoundaryConfig()
    warnings: list[str] = []

    matches = _find_matches(
        blocks, _build_prefixed_pattern(config), "prefixed", skip_list_items=False
    )
    if not matches and config.allow_numeric_fallback:
        matches = _find_matches(blocks, NUMERIC_PATTERN, "numeric", skip_list_items=True)
        if matches:
            warnings.append(
                f"No prefixed question labels (Q1, S2, …) found; fell back to plain "
                f"numbering and detected {len(matches)} questions. Verify the split."
            )

    if not matches:
        warnings.append("No question boundaries detected in this document.")
        if not blocks:
            return [], warnings
        return [_preamble_segment(blocks, len(blocks))], warnings

    boundaries: list[QuestionBoundary] = []
    first_start = matches[0].block_index
    if first_start > 0:
        boundaries.append(_preamble_segment(blocks, first_start))

    by_index = {block.index: block for block in blocks}
    ordered_indices = [block.index for block in blocks]

    for position, match in enumerate(matches):
        is_last = position == len(matches) - 1
        stop = len(ordered_indices) if is_last else ordered_indices.index(
            matches[position + 1].block_index
        )
        start = ordered_indices.index(match.block_index)
        segment_indices = ordered_indices[start:stop]

        label_block = by_index[match.block_index]
        title_runs = trim_runs_prefix(label_block.runs, match.end_offset)

        boundaries.append(
            QuestionBoundary(
                label=match.label,
                raw_label=match.raw_label,
                block_indices=segment_indices,
                start_index=segment_indices[0],
                end_index=segment_indices[-1],
                title_block_index=match.block_index,
                title_runs=title_runs,
                title_text=collapse_whitespace(runs_to_text(title_runs)),
                pattern=match.pattern,
            )
        )

    _reassign_trailing_headers(boundaries, by_index)
    warnings.extend(_duplicate_label_warnings(boundaries))
    return boundaries, warnings


def _reassign_trailing_headers(boundaries: list[QuestionBoundary], by_index: dict) -> None:
    """Move a trailing "ASK ALL, SC" header onto the question it introduces.

    The header sits above its own question's label, so plain segmentation
    leaves it at the tail of the *previous* question — which is how a type
    marker for Q1.3 ended up eligible to become an option of Q1.2. This is a
    question-boundary decision, not a decision about what the line means: the
    classifier still judges its role.
    """
    for position in range(len(boundaries) - 1):
        current, following = boundaries[position], boundaries[position + 1]

        moved: list[int] = []
        while len(current.block_indices) > 1:
            index = current.block_indices[-1]
            block = by_index.get(index)
            if block is None or not isinstance(block, ParagraphBlock):
                break
            if not block.text.strip():
                # An empty paragraph between the header and the next label.
                current.block_indices.pop()
                moved.insert(0, index)
                continue
            if not QUESTION_HEADER.match(block.text):
                break
            current.block_indices.pop()
            moved.insert(0, index)

        if not moved:
            continue

        following.block_indices[:0] = moved
        current.end_index = current.block_indices[-1]
        following.start_index = following.block_indices[0]


def _preamble_segment(blocks: list[Block], stop: int) -> QuestionBoundary:
    indices = [block.index for block in blocks[:stop]]
    return QuestionBoundary(
        label=None,
        block_indices=indices,
        start_index=indices[0],
        end_index=indices[-1],
        is_preamble=True,
    )


def _duplicate_label_warnings(boundaries: list[QuestionBoundary]) -> list[str]:
    seen: dict[str, int] = {}
    for boundary in boundaries:
        if boundary.label:
            seen[boundary.label] = seen.get(boundary.label, 0) + 1
    return [
        f"Duplicate question label '{label}' appears {count} times."
        for label, count in seen.items()
        if count > 1
    ]
