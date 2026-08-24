"""Text cleanup, ported from the team's `decipher-subl.py` default branch.

Word inserts typographic characters that Decipher's downstream tooling handles
badly, so they are normalised to plain ASCII on the way out.
"""

from __future__ import annotations

import re

from app.models.document import TextRun

#: Typographic characters Word substitutes automatically.
_ASCII_SUBSTITUTIONS = {
    "‘": "'",   # left single quote
    "’": "'",   # right single quote / apostrophe
    "‚": "'",
    "‛": "'",
    "“": '"',   # left double quote
    "”": '"',   # right double quote
    "„": '"',
    "–": "-",   # en dash
    "—": "-",   # em dash
    "―": "-",
    "−": "-",   # minus sign
    "…": "...",  # ellipsis
    " ": " ",   # non-breaking space
    " ": " ",
    " ": " ",
    "​": "",    # zero-width space
    "﻿": "",
    "‑": "-",   # non-breaking hyphen
}

#: An ``&`` that already begins a character or numeric entity.
_EXISTING_ENTITY = re.compile(r"&(?:[A-Za-z][A-Za-z0-9]{1,31}|#\d{1,7}|#[xX][0-9A-Fa-f]{1,6});")


def to_ascii(text: str) -> str:
    """Replace curly quotes, dashes and ellipses with plain ASCII."""
    for source, target in _ASCII_SUBSTITUTIONS.items():
        text = text.replace(source, target)
    return text


def escape_xml(text: str) -> str:
    """Escape a bare ``&`` without double-escaping an existing entity.

    ``Q&A`` becomes ``Q&amp;A``, but an already-correct ``&amp;`` or ``&#233;``
    is left alone — running the generator over previously escaped text must not
    corrupt it.
    """
    out: list[str] = []
    position = 0
    for match in _EXISTING_ENTITY.finditer(text):
        out.append(text[position:match.start()].replace("&", "&amp;"))
        out.append(match.group(0))
        position = match.end()
    out.append(text[position:].replace("&", "&amp;"))

    escaped = "".join(out)
    return escaped.replace("<", "&lt;").replace(">", "&gt;")


def clean(text: str) -> str:
    """Full cleanup: ASCII normalisation, entity-safe escaping, tidy spacing."""
    return re.sub(r"[ \t]+", " ", escape_xml(to_ascii(text))).strip()


def runs_to_markup(runs: list[TextRun]) -> str:
    """Render formatting runs as the markup Decipher accepts inside content.

    The workflow document is explicit that the generator — not the AI — turns
    Phase 1's runs into markup. Text is escaped first so that a literal ``<``
    in the questionnaire cannot become a tag.
    """
    parts: list[str] = []
    for run in runs:
        body = escape_xml(to_ascii(run.text))
        if not body.strip():
            parts.append(body)
            continue
        if run.bold:
            body = f"<b>{body}</b>"
        if run.italic:
            body = f"<i>{body}</i>"
        parts.append(body)
    return re.sub(r"[ \t]+", " ", "".join(parts)).strip()


def option_markup(raw_text: str, bold: bool = False, italic: bool = False) -> str:
    """Render one option/row/column line with its line-level formatting."""
    body = clean(raw_text)
    if not body:
        return body
    if bold:
        body = f"<b>{body}</b>"
    if italic:
        body = f"<i>{body}</i>"
    return body
