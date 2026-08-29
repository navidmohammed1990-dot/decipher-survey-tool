"""Rejoining an option whose text wrapped onto a second physical line.

A questionnaire written in Word wraps long option text, and copying it out
turns one logical option into two lines with the code stranded on the second:

    Buy it instead of another [BRAND] [FORMAT OF INTEREST]
    product you usually buy 1

This is a text-reflow artifact, the same class of problem as a table row split
across cells — not a judgment about what any line means. Roles are still the
classifier's to decide; this only repairs the lines it is shown.

Deliberately narrow. It merges only where the block itself establishes that
codes are the convention and the evidence is unambiguous, because wrongly
gluing a comment onto an option is worse than leaving a wrap unrepaired.
"""

from __future__ import annotations

import re

from app.classify.features import extract_features
from app.classify.lines import SourceLine
from app.models.document import TextRun

#: A bare code: a short number after a single space at the end of a line.
#: Too weak a signal to trust generally — "Aged 18 to 24" ends in a number too
#: — so it is read only inside a block that already codes its options with a
#: clear separator, and only on a line that could complete a wrap.
_BARE_TRAILING_CODE = re.compile(r"^(?P<text>.*\S)[ \t](?P<code>\d{1,3})[ \t]*$")

#: One clearly coded line is enough to establish that this list uses codes.
MIN_CODED_LINES = 1

#: Sentence-ending punctuation. A line closing with one of these is a finished
#: thought — an instruction or a title — not the first half of an option.
_SENTENCE_ENDINGS = (".", "?", ":", ";", "!")

#: A wrapped fragment is a fragment; a long line is more likely real content.
MAX_FRAGMENT_CHARS = 120

#: A line wraps because it filled the available width, so half an option is
#: long *for its block* - a heading or a variable name is short. Measured
#: against the block's own widest line rather than a fixed character count:
#: an absolute 25 was calibrated against "S2_AGE BANDS" and one long option,
#: and duly mis-read the first narrow table column it met, where every option
#: wraps well under 25 characters.
FRAGMENT_WIDTH_RATIO = 0.5

#: A variable name or an all-caps heading stands alone. Prose does not shout.
_SHOUTED = re.compile(r"^[A-Z0-9][A-Z0-9 _/&.\-]*$")


def _can_continue(line: SourceLine, min_chars: float) -> bool:
    """Whether this line could be the first half of a wrapped option."""
    text = line.text.strip()
    if not (min_chars <= len(text) <= MAX_FRAGMENT_CHARS):
        return False
    if _SHOUTED.match(text):
        return False
    if text.endswith(_SENTENCE_ENDINGS):
        return False
    if line.features.trailing_numeric_code or line.features.has_leading_enumeration:
        return False
    if "|" in text:
        # Normalisation only inserts a pipe where it found columns, so this
        # line already has structure — a code, a per-row note — and is
        # therefore complete rather than half of something.
        return False
    # Programmer notes and type markers stand alone.
    return not (line.features.matches_routing_keyword or line.features.matches_type_tag_pattern)


def _bare_code(text: str) -> tuple[str, str] | None:
    match = _BARE_TRAILING_CODE.match(text.strip())
    return (match.group("text"), match.group("code")) if match else None


def _completes(line: SourceLine) -> bool:
    """Whether this line ends an option, carrying the code that closes it."""
    if line.features.trailing_numeric_code:
        return True
    return _bare_code(line.text) is not None


def merge_wrapped_options(lines: list[SourceLine]) -> list[SourceLine]:
    """Rejoin option text split across two lines, renumbering the result.

    The first line of a block is never merged: it is the question's own text,
    and a title running onto a second line is not an option.
    """
    # Only explicitly separated codes establish the convention; bare trailing
    # numbers are then readable as codes, but never on their own evidence.
    explicit = sum(1 for line in lines if line.features.trailing_numeric_code)
    if len(lines) < 3 or explicit < MIN_CODED_LINES:
        return lines

    # The block's own widest answer line stands in for the width the source
    # wrapped at. The question's own text is excluded: a long stem would set
    # the bar above every option under it.
    widest = max(len(line.text.strip()) for line in lines[1:])
    min_chars = widest * FRAGMENT_WIDTH_RATIO

    merged: list[SourceLine] = []
    skip_next = False

    for position, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue

        following = lines[position + 1] if position + 1 < len(lines) else None
        if (
            position > 0
            and following is not None
            and _can_continue(line, min_chars)
            and _completes(following)
        ):
            merged.append(_join(line, following))
            skip_next = True
        else:
            merged.append(line)

    if len(merged) == len(lines):
        return lines

    # Indices are the classifier's addressing scheme, so they must stay dense.
    return [line.model_copy(update={"index": index}) for index, line in enumerate(merged)]


def _join(first: SourceLine, second: SourceLine) -> SourceLine:
    """One option from two lines, re-observing the joined text.

    A bare trailing code is rewritten into the separated form so the ordinary
    code extraction downstream sees it, rather than leaving the number glued
    to the option's wording.
    """
    text = f"{first.text.rstrip()} {second.text.lstrip()}"
    if not second.features.trailing_numeric_code:
        split = _bare_code(text)
        if split:
            text = f"{split[0]} | {split[1]}"

    runs = [*first.runs, TextRun(text=" "), *second.runs]

    return first.model_copy(
        update={
            "text": text,
            "runs": runs,
            "literal_marker": first.literal_marker or second.literal_marker,
            "marker": first.marker or second.marker,
            "features": extract_features(text, runs, is_table_row=second.features.is_table_row),
        }
    )
