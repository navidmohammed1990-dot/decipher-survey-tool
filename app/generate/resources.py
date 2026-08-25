"""Resource tag catalog and the deterministic element -> tag mapping.

Decipher resolves ``${res.X}`` at survey runtime, so the generator emits the
literal reference and never the text behind it. The catalog is only read so
the review UI can show a programmer what they are picking.

The catalog is parsed from the reference template rather than hardcoded here:
when the team edits a snippet, this tool follows without a code change.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

#: Matches a <res> entry anywhere in a file, including values containing markup
#: such as <br />. Non-greedy with DOTALL so multi-line values survive.
RES_PATTERN = re.compile(r'<res label="([^"]+)">(.*?)</res>', re.DOTALL)

#: Subject types a grid's rows can describe.
SUBJECT_TYPES: tuple[str, ...] = ("brand", "category", "product", "statement", "none")

#: Fallback when a grid's subject is unclear — the most generic wording.
DEFAULT_SUBJECT_TYPE = "statement"

#: (element, subject_type) -> resource label. Deterministic, never AI-chosen.
_GRID_SUBJECT_TAGS = {
    "brand": "Brand",
    "category": "Category",
    "product": "Product",
    "statement": "Statement",
    "none": "Statement",
}

#: Elements whose comment is a fixed tag regardless of subject.
_FLAT_TAGS = {
    "radio": "SR",
    "checkbox": "MR",
    "textarea": "Open",
    "text": "Open",
    "select": "Ranking",
    # html takes no comment at all.
    "html": None,
    # No dedicated numeric snippet exists in the template, so numeric entry
    # shares the open-response wording. SR was plainly wrong: it is
    # single-response radio wording on a free-entry field.
    "number": "Open",
}

#: Comments with no resource tag in the catalog, kept as literal text. Empty
#: now that number maps to Open; add entries here only for elements the
#: template has no snippet for.
LITERAL_COMMENT_DEFAULTS: dict[str, str] = {}


def resource_tag_for(element: str, subject_type: str = "none") -> str | None:
    """The resource label a question should reference, or ``None`` for no tag."""
    if element in ("radio_grid", "checkbox_grid"):
        prefix = "SR" if element == "radio_grid" else "MR"
        suffix = _GRID_SUBJECT_TAGS.get(subject_type, "Statement")
        return f"{prefix}{suffix}"
    return _FLAT_TAGS.get(element)


def parse_resource_catalog(text: str) -> dict[str, str]:
    """Extract ``label -> text`` from every ``<res>`` entry in ``text``."""
    return {label: value.strip() for label, value in RES_PATTERN.findall(text)}


def load_resource_catalog(path: str | Path | None = None) -> dict[str, str]:
    """Read the catalog from the reference template.

    A missing or unreadable template is not fatal: tag *selection* is pure
    logic and still works, only the preview text in the review UI is lost.
    """
    source = Path(path or settings.template_xml)
    try:
        catalog = parse_resource_catalog(source.read_text(encoding="utf-8", errors="replace"))
    except OSError as exc:
        logger.warning("Could not read resource template %s: %s", source, exc)
        return {}

    if not catalog:
        logger.warning("No <res> entries found in %s", source)
    return catalog


@lru_cache(maxsize=1)
def resource_catalog() -> dict[str, str]:
    """The catalog, read once at first use."""
    return load_resource_catalog()


def resource_text(label: str) -> str | None:
    return resource_catalog().get(label)


def reference(label: str) -> str:
    """The literal reference the generator emits."""
    return f"${{res.{label}}}"
