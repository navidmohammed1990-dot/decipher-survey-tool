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

#: A question label is letters then digits: Q1, QD24, MP2, APP1, D5A.
#:
#: Deliberately a shape rather than a list of known prefixes. An enumerated
#: whitelist only ever covers the questionnaires it was written against, and
#: silently drops the next house style it meets — QD24, P1 and MP2 all failed
#: that way. Anything of this shape is a label, whatever its letters spell.
LABEL_SHAPE = r"""
    [A-Za-z]{{1,{max_letters}}}      # QD, MP, APP - any letters, not a fixed set
    \d{{1,{max_digits}}}             # ...followed immediately by digits
    [A-Za-z]?                       # an optional sub-letter: D5A, Q10a
    (?:[._-]\d{{1,{max_digits}}})?   # an optional numeric sub-part: Q1.2, Q5_1
"""

@dataclass
class BoundaryConfig:
    """Tuning knobs for boundary detection."""

    allow_numeric_fallback: bool = True
    """When no lettered label exists anywhere, treat "1." / "2)" paragraphs as
    question starts."""
    max_label_digits: int = 3
    max_label_letters: int = 3
    prefixes: tuple[str, ...] | None = None
    """Optional restriction to specific prefixes.

    ``None`` accepts any label of the right shape, which is the point. Set it
    only to deliberately narrow detection for one awkward document.
    """
    extra_patterns: tuple[str, ...] = field(default_factory=tuple)


def _build_prefixed_pattern(config: BoundaryConfig) -> re.Pattern:
    """Compile the label matcher.

    Note there is no whitespace allowed between the letters and the digits.
    Permitting it would make "Yes 1." and "No 2." — ordinary coded options —
    look like question labels, which is a far worse failure than missing the
    rare "Q 12." written with a space.
    """
    shape = LABEL_SHAPE.format(
        max_letters=config.max_label_letters, max_digits=config.max_label_digits
    )
    if config.prefixes:
        alternation = "|".join(re.escape(p) for p in sorted(config.prefixes, key=len, reverse=True))
        shape = rf"(?:{alternation})\d{{1,{config.max_label_digits}}}[A-Za-z]?"

    return re.compile(
        rf"""^[ \t]*
        \[?[ \t]*
        (?P<label>{shape})
        [ \t]*\]?
        (?P<sep>[.):\-–—]+[ \t]*|[ \t]+|$)
        """,
        re.VERBOSE,
    )


#: Fallback for questionnaires that number questions without a letter prefix.
NUMERIC_PATTERN = re.compile(r"^[ \t]*(?P<label>\d{1,3})(?P<sep>[.)][ \t]*)")

#: An "ASK ..." opener. Narrow on purpose: it names the audience for the
#: question that FOLLOWS. Other routing keywords do not - a "TERMINATE IF"
#: after a question's options belongs to that question - so they stay put.
_ASK_OPENER = re.compile(
    r"^[ \t]*[\[\(]?[ \t]*ASK[ \t]+(?:ALL|IF|ONLY[ \t]+IF)\b",
    re.IGNORECASE,
)

#: Brackets and punctuation a marker may be dressed in. A marker introducing
#: the next question sits alone on its line; a sentence that merely contains
#: "SC" does not.
_MARKER_DRESSING = re.compile(r"[\[\]\(\),;:.\-\t ]")


def is_question_header(text: str) -> bool:
    """Whether this line introduces the question that follows it.

    A house-style header - "ASK ALL, SC", "[MC]", "SR PER ROW" - sits *above*
    its own question's label, so plain segmentation leaves it at the tail of
    the previous question.

    Which markers count is asked of :func:`detect_type_tag`, the one place that
    knows them, rather than answered by a second list here. The second list is
    how "SR" and "MR" came to be stranded: they were added as type tags and
    never added here, so a grid's own header stayed on the question above it.
    """
    from app.classify.features import detect_type_tag_span

    if _ASK_OPENER.match(text):
        return True
    found = detect_type_tag_span(text)
    if found is None:
        return False
    _, start, end = found
    return not _MARKER_DRESSING.sub("", text[:start] + text[end:]).strip()


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
        matches = _recurring_matches(blocks)
        if matches:
            warnings.append(
                f"No label pattern the tool knows; split on a prefix repeated at "
                f"the start of {len(matches)} paragraphs, which reads as this "
                f"document's label style. Verify the split."
            )

    if not matches:
        if not blocks:
            warnings.append("No question boundaries detected in this document.")
            return [], warnings
        segments = _gap_segments(blocks)
        if len(segments) > 1:
            warnings.append(
                f"No question label the tool could read; split on blank lines "
                f"instead and found {len(segments)} question(s), given placeholder "
                f"labels. Please confirm/correct each label."
            )
            return segments, warnings
        warnings.append("No question boundaries detected in this document.")
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
            if not is_question_header(block.text):
                break
            current.block_indices.pop()
            moved.insert(0, index)

        if not moved:
            continue

        following.block_indices[:0] = moved
        current.end_index = current.block_indices[-1]
        following.start_index = following.block_indices[0]


def _recurring_matches(blocks: list[Block]) -> list[_Match]:
    """Question starts found from a prefix the document repeats.

    Shares its judgment with the pasted-text path: a house style announces
    itself by recurring, so a token opening several paragraphs is this
    document's label whatever it spells. Without this a DOCX using an unknown
    prefix and no empty paragraphs between questions yields no questions at
    all - the whole file becomes one preamble.
    """
    from app.classify.paste import label_candidate, recurring_prefix

    paragraphs = [
        block
        for block in blocks
        if isinstance(block, ParagraphBlock) and not block.is_empty
    ]
    prefix = recurring_prefix([[block.text for block in paragraphs]])
    if prefix is None:
        return []

    matches: list[_Match] = []
    for block in paragraphs:
        found = label_candidate(block.text)
        if found and found[0] == prefix:
            _, label, end = found
            matches.append(
                _Match(
                    block_index=block.index,
                    label=label,
                    raw_label=block.text[:end].strip(),
                    end_offset=end,
                    pattern="recurring",
                )
            )
    return matches


def _gap_segments(blocks: list[Block]) -> list[QuestionBoundary]:
    """Split on blank paragraphs when no label pattern matched.

    A blank line between questions is the one separator every questionnaire
    uses, whatever it calls its questions. Without this a house style the
    label regex has never met - Sent1, Sent1B, Sent2 - makes the whole
    document one undifferentiated preamble and nothing gets classified.
    """
    groups: list[list[Block]] = []
    current: list[Block] = []
    for block in blocks:
        if isinstance(block, ParagraphBlock) and block.is_empty:
            if current:
                groups.append(current)
                current = []
            continue
        current.append(block)
    if current:
        groups.append(current)

    segments: list[QuestionBoundary] = []
    for position, group in enumerate(groups, start=1):
        head = group[0]
        runs = head.runs if isinstance(head, ParagraphBlock) else []
        indices = [block.index for block in group]
        segments.append(
            QuestionBoundary(
                label=f"Q{position}",
                block_indices=indices,
                start_index=indices[0],
                end_index=indices[-1],
                title_block_index=head.index if isinstance(head, ParagraphBlock) else None,
                title_runs=list(runs),
                title_text=collapse_whitespace(runs_to_text(runs)),
                pattern="gap",
            )
        )
    return segments


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


def match_question_label(
    text: str, config: BoundaryConfig | None = None, *, allow_numeric: bool = False
) -> tuple[str, str, int] | None:
    """Match a question label at the start of ``text``.

    Returns ``(normalised_label, raw_match, end_offset)``. Shared with the
    pasted-text splitter so both entry points recognise labels identically —
    a second implementation would drift from this one.
    """
    config = config or BoundaryConfig()
    prepared = normalize_for_matching(text)

    match = _build_prefixed_pattern(config).match(prepared)
    if match is None and allow_numeric:
        match = NUMERIC_PATTERN.match(prepared)
    if match is None:
        return None
    return normalize_label(match.group("label")), match.group(0).strip(), match.end()
