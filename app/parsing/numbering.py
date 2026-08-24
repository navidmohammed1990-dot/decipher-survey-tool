"""Resolution of Word list numbering into rendered marker text.

An auto-numbered option list stores nothing in the paragraph text — the "1.",
"2.", "3." a programmer sees on screen live in numbering.xml. Losing them would
mean losing the option values, so this module replays Word's counters as the
document is walked in order.
"""

from __future__ import annotations

from docx.oxml.ns import qn

_ROMAN = [
    (1000, "m"), (900, "cm"), (500, "d"), (400, "cd"),
    (100, "c"), (90, "xc"), (50, "l"), (40, "xl"),
    (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i"),
]


def _to_roman(value: int) -> str:
    if value <= 0:
        return str(value)
    out = []
    for amount, numeral in _ROMAN:
        count, value = divmod(value, amount)
        out.append(numeral * count)
    return "".join(out)


def _to_letter(value: int) -> str:
    """1 -> a, 26 -> z, 27 -> aa, matching Word's alphabetic numbering."""
    if value <= 0:
        return str(value)
    out = ""
    while value > 0:
        value, rem = divmod(value - 1, 26)
        out = chr(ord("a") + rem) + out
    return out


def format_counter(value: int, num_fmt: str | None) -> str:
    match num_fmt:
        case "lowerLetter":
            return _to_letter(value)
        case "upperLetter":
            return _to_letter(value).upper()
        case "lowerRoman":
            return _to_roman(value)
        case "upperRoman":
            return _to_roman(value).upper()
        case "decimalZero":
            return f"{value:02d}"
        case "none":
            return ""
        case _:
            return str(value)


class NumberingResolver:
    """Replays list counters across a document to render list markers.

    One instance corresponds to one pass over one document; counters are
    stateful, so walking the body twice with the same resolver would continue
    the numbering rather than restart it.
    """

    def __init__(self, document):
        self._levels: dict[tuple[int, int], dict] = {}
        self._counters: dict[tuple[int, int], int] = {}
        self._load(document)

    # -- numbering.xml ----------------------------------------------------

    def _load(self, document) -> None:
        try:
            numbering = document.part.numbering_part.element
        except (AttributeError, KeyError, NotImplementedError):
            # A document with no lists has no numbering part at all.
            return

        abstract: dict[int, dict[int, dict]] = {}
        for abstract_num in numbering.findall(qn("w:abstractNum")):
            abstract_id = _int_attr(abstract_num, "w:abstractNumId")
            if abstract_id is None:
                continue
            abstract[abstract_id] = self._read_levels(abstract_num)

        for num in numbering.findall(qn("w:num")):
            num_id = _int_attr(num, "w:numId")
            if num_id is None:
                continue
            ref = num.find(qn("w:abstractNumId"))
            abstract_id = _int_attr(ref, "w:val") if ref is not None else None
            levels = dict(abstract.get(abstract_id, {}))

            # w:lvlOverride lets one list restart another's numbering.
            for override in num.findall(qn("w:lvlOverride")):
                level = _int_attr(override, "w:ilvl")
                if level is None:
                    continue
                merged = dict(levels.get(level, {}))
                start_override = override.find(qn("w:startOverride"))
                if start_override is not None:
                    merged["start"] = _int_attr(start_override, "w:val") or 1
                lvl = override.find(qn("w:lvl"))
                if lvl is not None:
                    merged.update(self._read_level(lvl))
                levels[level] = merged

            for level, spec in levels.items():
                self._levels[(num_id, level)] = spec

    def _read_levels(self, abstract_num) -> dict[int, dict]:
        levels = {}
        for lvl in abstract_num.findall(qn("w:lvl")):
            level = _int_attr(lvl, "w:ilvl")
            if level is not None:
                levels[level] = self._read_level(lvl)
        return levels

    def _read_level(self, lvl) -> dict:
        def val(tag: str):
            el = lvl.find(qn(tag))
            return el.get(qn("w:val")) if el is not None else None

        start = val("w:start")
        return {
            "start": int(start) if start and start.isdigit() else 1,
            "num_fmt": val("w:numFmt"),
            "lvl_text": val("w:lvlText"),
        }

    # -- counters ---------------------------------------------------------

    def next_marker(self, num_id: int, level: int) -> tuple[str | None, str | None]:
        """Advance the counter for ``(num_id, level)`` and render its marker.

        Returns ``(marker, num_fmt)``. Deeper levels reset, matching Word.
        """
        spec = self._levels.get((num_id, level), {})
        num_fmt = spec.get("num_fmt")

        if num_fmt == "bullet":
            # Bullets have no counter; the glyph is the literal lvlText.
            return (spec.get("lvl_text") or "•"), num_fmt

        key = (num_id, level)
        if key in self._counters:
            self._counters[key] += 1
        else:
            self._counters[key] = spec.get("start", 1)

        for (other_id, other_level) in list(self._counters):
            if other_id == num_id and other_level > level:
                del self._counters[(other_id, other_level)]

        lvl_text = spec.get("lvl_text") or "%{}.".format(level + 1)
        marker = self._render(lvl_text, num_id, level)
        return marker, num_fmt

    def _render(self, lvl_text: str, num_id: int, level: int) -> str:
        """Substitute ``%1``, ``%2`` … in a level's text with live counters."""
        out = lvl_text
        for placeholder_level in range(level, -1, -1):
            token = f"%{placeholder_level + 1}"
            if token not in out:
                continue
            spec = self._levels.get((num_id, placeholder_level), {})
            value = self._counters.get((num_id, placeholder_level), spec.get("start", 1))
            out = out.replace(token, format_counter(value, spec.get("num_fmt")))
        return out


def _int_attr(element, attr: str) -> int | None:
    if element is None:
        return None
    raw = element.get(qn(attr))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def paragraph_numbering(paragraph) -> tuple[int, int, bool] | None:
    """Return ``(num_id, level, from_style)`` for a list paragraph.

    Numbering can be applied directly to the paragraph or inherited from its
    style; direct formatting wins.
    """
    num_pr = _find_num_pr(paragraph._p.find(qn("w:pPr")))
    if num_pr is not None:
        num_id = _int_attr(num_pr.find(qn("w:numId")), "w:val")
        if num_id:
            level = _int_attr(num_pr.find(qn("w:ilvl")), "w:val") or 0
            return num_id, level, False

    style = paragraph.style
    seen: set[int] = set()
    while style is not None and id(style) not in seen:
        seen.add(id(style))
        num_pr = _find_num_pr(style.element.find(qn("w:pPr")))
        if num_pr is not None:
            num_id = _int_attr(num_pr.find(qn("w:numId")), "w:val")
            if num_id:
                level = _int_attr(num_pr.find(qn("w:ilvl")), "w:val") or 0
                return num_id, level, True
        style = style.base_style
    return None


def _find_num_pr(ppr):
    return None if ppr is None else ppr.find(qn("w:numPr"))
