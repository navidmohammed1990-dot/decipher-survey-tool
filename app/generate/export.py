"""Assembling generated fragments into a document, and checking them.

Fragments use the ``atm1d:`` and ``ss:`` prefixes without declaring them, which
is correct — the namespaces belong on the survey root. That does mean a bare
fragment is not standalone-parseable, so both the validator and the download
wrap the fragments in a root first.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

from app.config import settings

#: Prefixes the generated fragments use without declaring.
NAMESPACE_PREFIXES = ("atm1d", "ss")


def namespace_declarations() -> str:
    return (
        f'xmlns:atm1d="{settings.xmlns_atm1d}" '
        f'xmlns:ss="{settings.xmlns_ss}"'
    )


def wrap_survey(fragments: str, *, name: str | None = None) -> str:
    """Wrap fragments in a minimal ``<survey>`` root that declares the prefixes.

    The root here is a placeholder so the export opens as valid XML. Replace it
    with the team's canonical survey root before running the file.
    """
    attrs = namespace_declarations()
    if name:
        safe = name.replace("&", "&amp;").replace('"', "&quot;")
        attrs = f'alt="{safe}" {attrs}'
    return f"<survey {attrs}>\n\n{fragments}\n\n</survey>\n"


@dataclass
class WellFormedResult:
    ok: bool
    error: str | None = None

    def __bool__(self) -> bool:
        return self.ok


def check_well_formed(fragments: str) -> WellFormedResult:
    """Parse the fragments inside a namespace-declaring root.

    Checks syntax only — element-level Decipher rules are a later phase.
    """
    try:
        ET.fromstring(wrap_survey(fragments))
    except ET.ParseError as exc:
        return WellFormedResult(ok=False, error=str(exc))
    return WellFormedResult(ok=True)
