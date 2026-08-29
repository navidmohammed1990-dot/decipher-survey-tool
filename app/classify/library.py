"""A correction library that survives a restart.

Session memory (:mod:`app.classify.corrections`) forgets everything when the
server stops, so a programmer who corrected the same pattern yesterday
corrected it again today. This stores confirmed corrections on disk.

Scope is preserved deliberately. Phase 7's rule still holds: one client's house
conventions must not leak into another's questionnaire, so a stored correction
is offered back only for the document it came from — unless someone marks it
``always``, which says the pattern is general rather than client-specific.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from app.classify.corrections import Correction

logger = logging.getLogger(__name__)

#: Included in a prompt at most this many, newest first. The library may grow
#: without bound; the prompt may not — every entry costs inference time.
MAX_PROMPTED = 3


class StoredCorrection(BaseModel):
    """One correction, with the scope that decides when it is offered back."""

    correction: Correction
    document: str = ""
    always: bool = False
    """True when the pattern is general, not particular to this document."""
    recorded_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = "review"
    """Which route recorded it: ``review``, ``quick`` or ``command``."""


class CorrectionLibrary:
    """Confirmed corrections, persisted as JSON."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._lock = threading.Lock()
        self._path = Path(path) if path else None
        self._entries: list[StoredCorrection] = []
        self._load()

    # -- persistence ------------------------------------------------------

    def _load(self) -> None:
        if self._path is None or not self._path.is_file():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # A damaged library must not stop the tool starting.
            logger.warning("Could not read correction library %s: %s", self._path, exc)
            return

        for item in raw if isinstance(raw, list) else []:
            try:
                self._entries.append(StoredCorrection.model_validate(item))
            except Exception:  # pragma: no cover - one bad row, not a crash
                logger.warning("Skipping unreadable correction entry")

    def _save_locked(self) -> None:
        if self._path is None:
            return
        payload = json.dumps([e.model_dump() for e in self._entries], indent=2)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Written via a temp file so an interrupted save cannot truncate
            # the library to nothing.
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self._path.parent, delete=False
            ) as handle:
                handle.write(payload)
                temporary = Path(handle.name)
            os.replace(temporary, self._path)
        except OSError as exc:
            logger.warning("Could not write correction library %s: %s", self._path, exc)

    # -- use --------------------------------------------------------------

    def record(
        self, correction: Correction, document: str = "", *,
        always: bool = False, source: str = "review",
    ) -> bool:
        """Store a correction if it actually changed the model's answer."""
        if not correction.is_meaningful():
            return False

        with self._lock:
            # One entry per question per document; a later edit supersedes.
            self._entries = [
                entry for entry in self._entries
                if not (entry.document == document and entry.correction.label == correction.label)
            ]
            self._entries.append(
                StoredCorrection(
                    correction=correction, document=document, always=always, source=source
                )
            )
            self._save_locked()
        return True

    def entries(self) -> list[StoredCorrection]:
        with self._lock:
            return list(self._entries)

    def for_document(self, document: str) -> list[Correction]:
        """Corrections that apply here: this document's, plus general ones."""
        with self._lock:
            return [
                entry.correction
                for entry in self._entries
                if entry.always or (document and entry.document == document)
            ]

    def promote(self, label: str, document: str = "") -> bool:
        """Mark a correction as general, so every document sees it."""
        with self._lock:
            for entry in self._entries:
                if entry.correction.label == label and (
                    not document or entry.document == document
                ):
                    entry.always = True
                    self._save_locked()
                    return True
        return False

    def prompt_prefix(self, document: str) -> str:
        """Few-shot examples for this document, oldest first and bounded."""
        applicable = self.for_document(document)[-MAX_PROMPTED:]
        if not applicable:
            return ""
        return "\n\n".join(c.as_prompt_example() for c in applicable) + "\n\n"

    def clear(self) -> None:
        with self._lock:
            self._entries = []
            self._save_locked()


_default: CorrectionLibrary | None = None


def default_library() -> CorrectionLibrary:
    """The process-wide library, created on first use.

    Built lazily rather than at import time: constructing it reads a file, and
    a module import should not touch the disk or care what else has loaded yet.
    """
    global _default
    if _default is None:
        from app.config import settings

        _default = CorrectionLibrary(settings.correction_library or None)
    return _default
