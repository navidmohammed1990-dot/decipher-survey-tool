"""In-memory store for the draft currently under review.

The tool is single-user and local by design, so the draft lives in process
memory and dies with the server. Persisting it is a later decision — the
workflow document puts storage at "local files initially, database later".
"""

from __future__ import annotations

import threading

from app.models.survey import Question, QuestionDraft


class DraftStore:
    """Holds the one draft the survey programmer is working on."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._draft = QuestionDraft()

    def get(self) -> QuestionDraft:
        with self._lock:
            return self._draft.model_copy(deep=True)

    def replace(self, draft: QuestionDraft) -> QuestionDraft:
        with self._lock:
            self._draft = draft.model_copy(deep=True)
            return self._draft.model_copy(deep=True)

    def update_question(self, label: str, updated: Question) -> Question | None:
        with self._lock:
            for position, existing in enumerate(self._draft.questions):
                if existing.label == label:
                    self._draft.questions[position] = updated.model_copy(deep=True)
                    return updated.model_copy(deep=True)
            return None

    def clear(self) -> None:
        with self._lock:
            self._draft = QuestionDraft()


draft_store = DraftStore()
