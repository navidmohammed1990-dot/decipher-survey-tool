"""Session-level learning from survey programmer corrections.

A questionnaire follows one house's conventions throughout. When the programmer
fixes the first question the model got wrong, that correction is the best
possible evidence for how the *rest of this document* should be read — so it is
fed back as a few-shot example for later questions in the same session.

Local only: no fine-tuning, no external calls, nothing persisted. Memory is
cleared between documents so one client's conventions never leak into another's.
"""

from __future__ import annotations

import json
import threading

from pydantic import BaseModel, Field

#: How many recent corrections to carry. Enough to convey a convention without
#: crowding out the question actually being classified.
MAX_CORRECTIONS = 3


class Correction(BaseModel):
    """One survey programmer correction, as an example for later calls."""

    label: str = ""
    original_lines: list[str] = Field(default_factory=list)
    ai_said: dict = Field(default_factory=dict)
    sp_corrected_to: dict = Field(default_factory=dict)

    def is_meaningful(self) -> bool:
        """True when the programmer actually changed the model's answer."""
        if not self.original_lines or not self.ai_said:
            return False
        return any(
            self.ai_said.get(key) != value
            for key, value in self.sp_corrected_to.items()
        )

    def as_prompt_example(self) -> str:
        return (
            "The survey programmer corrected an earlier question in this document:\n"
            f"Lines: {json.dumps(self.original_lines[:12])}\n"
            f"You classified it as: {json.dumps(self.ai_said, sort_keys=True)}\n"
            f"It should have been: {json.dumps(self.sp_corrected_to, sort_keys=True)}\n"
            "Learn from this pattern for the rest of this document."
        )


class CorrectionMemory:
    """The corrections made so far on the current document.

    Backed by a library when one is attached, so switching back to a document
    restores what was already corrected on it rather than starting over.
    """

    def __init__(self, library=None, *, persist: bool = False) -> None:
        self._lock = threading.Lock()
        self._corrections: list[Correction] = []
        self._document: str | None = None
        self._library = library
        self._persist = persist
        self._source = "review"

    def _store(self):
        """The library to write through to, if any.

        An explicit library always wins, which is what lets tests swap in an
        isolated one; otherwise the shared on-disk library is resolved lazily.
        """
        if self._library is not None:
            return self._library
        if not self._persist:
            return None
        from app.classify.library import default_library

        return default_library()

    def record(self, correction: Correction) -> bool:
        """Keep a correction if it actually changed something."""
        if not correction.is_meaningful():
            return False
        store = self._store()
        if store is not None:
            store.record(correction, self._document or "", source=self._source)
        with self._lock:
            # One entry per question: a second edit supersedes the first.
            self._corrections = [c for c in self._corrections if c.label != correction.label]
            self._corrections.append(correction)
            del self._corrections[:-MAX_CORRECTIONS]
        return True

    def recent(self) -> list[Correction]:
        with self._lock:
            return list(self._corrections)

    def prompt_prefix(self) -> str:
        """Few-shot examples to prepend to the system prompt, oldest first."""
        corrections = self.recent()
        if not corrections:
            return ""
        return "\n\n".join(c.as_prompt_example() for c in corrections) + "\n\n"

    def use_document(self, name: str | None) -> None:
        """Switch documents, discarding corrections from the previous one.

        Anything the library already holds for this document comes back, so a
        restart mid-review does not lose the corrections already made.
        """
        with self._lock:
            if name == self._document:
                return
            self._document = name
            self._corrections = []
            store = self._store()
            if store is not None and name:
                self._corrections = store.for_document(name)[-MAX_CORRECTIONS:]

    def clear(self) -> None:
        with self._lock:
            self._corrections = []
            self._document = None


correction_memory = CorrectionMemory(persist=True)
