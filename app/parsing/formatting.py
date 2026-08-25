"""Character-level formatting extraction.

The workflow document is explicit that formatting must come from the document
parser and never be inferred by the AI, so this module resolves bold/italic the
way Word itself does: direct run formatting first, then the character style
chain, then the paragraph style chain, then the document defaults.
"""

from __future__ import annotations

from docx.oxml.ns import qn
from docx.text.run import Run

from app.models.document import TextRun

#: Attributes of ``docx.text.run.Font`` we resolve, mapped to run element tags.
_FORMAT_ATTRS = {"bold": "w:b", "italic": "w:i", "underline": "w:u"}

#: Word writes "auto" when a run follows the document default colour.
_AUTO_COLOR = "auto"

#: Tracked deletions carry text that is not part of the current document.
_SKIPPED_CONTAINERS = frozenset({qn("w:del")})


def _iter_run_elements(element):
    """Yield ``w:r`` elements in document order, descending into containers.

    Runs live directly under ``w:p`` but also inside hyperlinks, smart tags,
    content controls and tracked insertions. ``Paragraph.runs`` only sees the
    direct children, which would silently drop hyperlinked question text.
    """
    for child in element:
        if child.tag in _SKIPPED_CONTAINERS:
            continue
        if child.tag == qn("w:r"):
            yield child
        else:
            yield from _iter_run_elements(child)


def _style_chain_value(style, attr: str):
    """Walk a style and its ``basedOn`` ancestors for the first set value."""
    seen: set[int] = set()
    while style is not None and id(style) not in seen:
        seen.add(id(style))
        value = getattr(style.font, attr, None)
        if value is not None:
            return value
        style = style.base_style
    return None


def _coerce_underline(value):
    """Normalise ``Font.underline`` to a bool.

    It may be ``True``, ``False``, ``None`` or a ``WD_UNDERLINE`` member;
    ``WD_UNDERLINE.NONE`` compares equal to 0 and means "not underlined".
    """
    if value is None or isinstance(value, bool):
        return value
    try:
        return int(value) != 0
    except (TypeError, ValueError):
        return True


def _style_chain_color(style) -> str | None:
    seen: set[int] = set()
    while style is not None and id(style) not in seen:
        seen.add(id(style))
        value = _color_of(getattr(style, "font", None))
        if value is not None:
            return value
        style = style.base_style
    return None


def _color_of(font) -> str | None:
    """Read a font's explicit RGB colour, if it has one.

    ``ColorFormat.rgb`` raises for theme colours and returns ``None`` when
    unset, so both are treated as "no explicit colour".
    """
    if font is None:
        return None
    try:
        color = font.color
        if color is None or color.rgb is None:
            return None
        return str(color.rgb).upper()
    except (AttributeError, ValueError, TypeError):
        return None


def resolve_run_color(run, paragraph) -> str | None:
    """Resolve a run's colour through the same inheritance chain as bold."""
    for candidate in (run.font, None):
        if candidate is not None:
            value = _color_of(candidate)
            if value is not None:
                return value
    for style in (run.style, paragraph.style):
        value = _style_chain_color(style)
        if value is not None:
            return value
    return None


def document_defaults(document) -> dict[str, bool]:
    """Read ``docDefaults/rPrDefault`` from styles.xml."""
    defaults = {attr: False for attr in _FORMAT_ATTRS}
    try:
        styles_el = document.styles.element
    except AttributeError:
        return defaults

    doc_defaults = styles_el.find(qn("w:docDefaults"))
    if doc_defaults is None:
        return defaults
    rpr_default = doc_defaults.find(qn("w:rPrDefault"))
    if rpr_default is None:
        return defaults
    rpr = rpr_default.find(qn("w:rPr"))
    if rpr is None:
        return defaults

    for attr, tag in _FORMAT_ATTRS.items():
        el = rpr.find(qn(tag))
        if el is None:
            continue
        val = el.get(qn("w:val"))
        if attr == "underline":
            defaults[attr] = val not in (None, "none", "0", "false")
        else:
            # A toggle element with no w:val attribute means "on".
            defaults[attr] = val not in ("0", "false", "off")
    return defaults


def resolve_run_format(
    run: Run, paragraph, defaults: dict[str, bool] | None = None
) -> dict[str, bool]:
    """Resolve bold/italic/underline for one run, following style inheritance."""
    defaults = defaults or {}
    resolved: dict[str, bool] = {}

    for attr in _FORMAT_ATTRS:
        value = getattr(run.font, attr, None)
        if value is None:
            value = _style_chain_value(run.style, attr)
        if value is None:
            value = _style_chain_value(paragraph.style, attr)
        if value is None:
            value = defaults.get(attr)

        if attr == "underline":
            value = _coerce_underline(value)
        resolved[attr] = bool(value)

    return resolved


def merge_runs(runs: list[TextRun]) -> list[TextRun]:
    """Collapse adjacent runs that share formatting.

    Word splits runs for reasons that have nothing to do with appearance —
    spell-check state, revision ids, language tags — so a single visually
    uniform sentence often arrives as a dozen runs.
    """
    merged: list[TextRun] = []
    for run in runs:
        if not run.text:
            continue
        if merged and merged[-1].formatting_key() == run.formatting_key():
            merged[-1].text += run.text
        else:
            merged.append(run.model_copy())
    return merged


def extract_runs(paragraph, defaults: dict[str, bool] | None = None) -> list[TextRun]:
    """Extract merged, formatting-resolved runs from a paragraph."""
    runs: list[TextRun] = []
    for element in _iter_run_elements(paragraph._p):
        run = Run(element, paragraph)
        text = run.text
        if not text:
            continue
        fmt = resolve_run_format(run, paragraph, defaults)
        runs.append(TextRun(text=text, color=resolve_run_color(run, paragraph), **fmt))
    return merge_runs(runs)


def runs_to_text(runs: list[TextRun]) -> str:
    return "".join(run.text for run in runs)


def trim_runs_prefix(runs: list[TextRun], n_chars: int) -> list[TextRun]:
    """Drop the first ``n_chars`` characters while preserving formatting.

    Used to strip a question label ("Q5. ") from the title without losing the
    formatting of the text that follows it.
    """
    if n_chars <= 0:
        return [run.model_copy() for run in runs]

    remaining = n_chars
    trimmed: list[TextRun] = []
    for run in runs:
        if remaining <= 0:
            trimmed.append(run.model_copy())
        elif remaining >= len(run.text):
            remaining -= len(run.text)
        else:
            kept = run.model_copy()
            kept.text = run.text[remaining:]
            remaining = 0
            trimmed.append(kept)
    return merge_runs(trimmed)
