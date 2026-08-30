"""Factual hints about a line, for the classifier to weigh as evidence.

Nothing in this module assigns a role. A red line is not thereby routing text;
a line with a leading number is not thereby an option. Patterns are only ever
as good as the questionnaire they were written against, so they observe and the
classifier judges — the next document may colour routing text differently, use
brackets instead of caps, or follow a house convention nobody has seen.

Everything here is computed once per line and passed along as context. It is
never used to filter, exclude, or pre-sort anything.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from app.models.document import TextRun
from app.parsing.normalize import detect_literal_marker

#: Openers that commonly introduce a note to the programmer. Presence is a
#: hint, not a verdict — "Please skip to the next section" is respondent-facing,
#: and an option can legitimately read "Randomly assigned by my employer".
ROUTING_KEYWORDS = (
    "ASK ALL",
    "ASK IF",
    "ASK ONLY IF",
    "TERMINATE",
    "SKIP TO",
    "GO TO",
    "QUALIFY IF",
    "RANDOMLY ASSIGN",
    "RANDOMIZE",
    "ROTATE",
    "PROGRAMMER NOTE",
    "INTERVIEWER NOTE",
    "SCRIPTER NOTE",
    "DP NOTE",
    "QUOTA",
)

#: The same openers as patterns, tolerating the inflections questionnaires use
#: ("RANDOMLY ASSIGNED", "QUALIFIES IF"). Matching inflections deliberately
#: widens the hint: a real option that reads like an instruction is precisely
#: what the disagreement net exists to surface.
_ROUTING_PATTERNS = (
    r"ASK\s+ALL",
    r"ASK\s+(?:ONLY\s+)?IF",
    r"TERMINATE[SD]?",
    r"SKIP\s+TO",
    r"GO\s+TO",
    r"QUALIF(?:Y|IES|IED)\s+IF",
    r"RANDOMLY\s+ASSIGN(?:ED|S)?",
    r"RANDOMI[SZ]E[SD]?",
    r"ROTATE[SD]?",
    r"(?:PROGRAMMER|PROG|INTERVIEWER|SCRIPTER|DP)\s+NOTE",
    r"QUOTAS?",
)

_ROUTING_RE = re.compile(
    r"^\s*[\[\(]?\s*(?:" + "|".join(_ROUTING_PATTERNS) + r")\b",
    re.IGNORECASE,
)

#: House type markers, e.g. "ASK ALL, SC" or "[MC]" or "SC:".
TYPE_TAGS = {
    "SC": "SC",
    "MC": "MC",
    "SR": "SR",
    "MR": "MR",
    "OE": "OE",
    "NUM": "NUM",
    "GRID": "GRID",
}

#: Case-sensitive on purpose. These markers are written in capitals, and
#: matching loosely would read the "Mr" in "Mr. Smith" as a type signal.
_TYPE_TAG_RE = re.compile(
    r"(?:^|[,;:\-\s\[\(])(" + "|".join(TYPE_TAGS) + r")(?:$|[,;:\-\s\]\)])"
)

#: A grid marker: one response per repeated thing. House styles write the same
#: idea a dozen ways - "SR PER ROW", "SR per statement", "MR per brand",
#: "SR for each item" - so this matches the *shape* of the idea rather than one
#: remembered phrase. Matching only "SR PER ROW" is what made an APP3 grid come
#: out as a flat radio. SR/MR stay case-sensitive so "Mr. Smith" is not a type
#: marker; the words around them are not.
_PER_ROW_RE = re.compile(r"\b(SR|MR)\b[\s,:;\-]*(?i:per|for\s+each|against\s+each)\s+\w+")

#: The same idea written without an SR/MR marker at all. Which kind of grid is
#: then the AI's call, so this only says "grid".
_GRID_PHRASE_RE = re.compile(
    r"\b(?:one|single|1)\s+\w*\s*(?:response|answer|selection|code)\s+"
    r"(?:per|for\s+each)\s+\w+",
    re.IGNORECASE,
)

#: A code the source gave an option, e.g. "Male | 1", "Other (97)", "None [99]".
_TRAILING_CODE_RE = re.compile(
    r"(?:\|\s*|\(\s*|\[\s*|\{\s*)(?P<code>\d{1,3})\s*[\)\]\}]?\s*$"
)

#: How a hex colour maps to a name a person would use. Coarse on purpose:
#: the classifier only needs "this is red-ish", not the exact shade.
_COLOR_NAMES: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("black", (0, 0, 0)),
    ("white", (255, 255, 255)),
    ("grey", (128, 128, 128)),
    ("red", (200, 0, 0)),
    ("orange", (230, 120, 0)),
    ("yellow", (220, 200, 0)),
    ("green", (0, 150, 60)),
    ("blue", (0, 80, 200)),
    ("purple", (130, 0, 180)),
)

#: Colours at or below this distance from black count as "not coloured".
_NEAR_BLACK = 60


class LineFeatures(BaseModel):
    """Observations about one line. No field here means "this line is an X"."""

    has_leading_enumeration: bool = False
    is_table_row: bool = False
    is_bold: bool = False
    is_struck: bool = False
    is_colored: bool = False
    color_hint: str | None = None
    trailing_numeric_code: str | None = None
    matches_routing_keyword: bool = False
    matches_type_tag_pattern: bool = False
    type_tag_value: str | None = None

    def as_prompt_hints(self) -> str:
        """Render for the model, listing only what was actually observed."""
        parts: list[str] = []
        if self.has_leading_enumeration:
            parts.append("has_leading_enumeration=true")
        if self.is_table_row:
            parts.append("is_table_row=true")
        if self.is_bold:
            parts.append("is_bold=true")
        if self.is_struck:
            parts.append("is_struck=true")
        if self.is_colored:
            parts.append(f"color_hint={self.color_hint or 'other'}")
        if self.trailing_numeric_code:
            parts.append(f"trailing_numeric_code={self.trailing_numeric_code}")
        if self.matches_routing_keyword:
            parts.append("matches_routing_keyword=true")
        if self.matches_type_tag_pattern:
            parts.append(f"matches_type_tag_pattern=true, type_tag_value={self.type_tag_value}")
        return ", ".join(parts) if parts else "none"


def _hex_to_rgb(value: str) -> tuple[int, int, int] | None:
    try:
        return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
    except (ValueError, IndexError):
        return None


def color_name(hex_value: str | None) -> str | None:
    """Nearest common colour name, or ``None`` for default/near-black text."""
    if not hex_value:
        return None
    rgb = _hex_to_rgb(hex_value)
    if rgb is None:
        return None

    def distance(other):
        return sum((a - b) ** 2 for a, b in zip(rgb, other)) ** 0.5

    if distance((0, 0, 0)) <= _NEAR_BLACK:
        return None

    name, _ = min(_COLOR_NAMES, key=lambda entry: distance(entry[1]))
    return None if name in ("black", "white") else name


#: A numeric range stated in prose: "min 0 max 200", "Min: 0, Max: 200",
#: "0 to 200". Questionnaires write the range as a directive beside the row
#: rather than as a field, so it has to be read out of the words.
_BOUNDS_RE = re.compile(
    r"\bmin(?:imum)?\b\D{0,4}(?P<min>\d+)\D{0,12}?\bmax(?:imum)?\b\D{0,4}(?P<max>\d+)",
    re.IGNORECASE,
)


def detect_numeric_bounds(text: str) -> tuple[str, str] | None:
    """The min and max a line states, if it states both.

    Only a pair counts. A lone "max 200" leaves the range open at one end,
    which is not something to guess a zero for.
    """
    match = _BOUNDS_RE.search(text)
    return (match.group("min"), match.group("max")) if match else None


def detect_trailing_code(text: str) -> str | None:
    """The code a source line gave an option, if it gave one.

    Requires a separator or bracket: a bare trailing number would make "Under
    18" look like code 18.
    """
    match = _TRAILING_CODE_RE.search(text)
    return match.group("code") if match else None


def strip_trailing_code(text: str) -> str:
    """The option's text without its code, e.g. ``Male | 1`` -> ``Male``."""
    return _TRAILING_CODE_RE.sub("", text).strip(" |-–—\t")


def detect_type_tag_span(text: str) -> tuple[str, int, int] | None:
    """An explicit type marker with the span it occupies, if the line has one.

    The span is what lets a caller ask "is this line *only* a marker?" without
    keeping its own copy of the marker vocabulary - which is how the boundary
    detector's list came to be missing SR and MR.
    """
    per_row = _PER_ROW_RE.search(text)
    if per_row:
        return f"{per_row.group(1)}_GRID", per_row.start(), per_row.end()

    match = _TYPE_TAG_RE.search(text)
    if match:
        return TYPE_TAGS[match.group(1).upper()], match.start(1), match.end(1)

    phrase = _GRID_PHRASE_RE.search(text)
    if phrase:
        return "GRID", phrase.start(), phrase.end()
    return None


def detect_type_tag(text: str) -> str | None:
    """An explicit SC/MC/SR/MR-style marker, if the line carries one."""
    found = detect_type_tag_span(text)
    return found[0] if found else None


def matches_routing_keyword(text: str) -> bool:
    return bool(_ROUTING_RE.match(text))


def extract_features(
    text: str, runs: list[TextRun] | None = None, *, is_table_row: bool = False
) -> LineFeatures:
    """Observe one line. Assigns no role and excludes nothing."""
    runs = runs or []
    meaningful = [run for run in runs if run.text.strip()]

    colors = {run.color for run in meaningful if run.color}
    hint = next((name for name in (color_name(c) for c in sorted(colors)) if name), None)

    type_tag = detect_type_tag(text)
    return LineFeatures(
        has_leading_enumeration=detect_literal_marker(text) is not None,
        is_table_row=is_table_row,
        is_bold=bool(meaningful) and all(run.bold for run in meaningful),
        is_struck=bool(meaningful) and all(run.strike for run in meaningful),
        is_colored=hint is not None,
        color_hint=hint,
        trailing_numeric_code=detect_trailing_code(text),
        matches_routing_keyword=matches_routing_keyword(text),
        matches_type_tag_pattern=type_tag is not None,
        type_tag_value=type_tag,
    )
