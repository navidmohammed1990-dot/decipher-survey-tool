"""In-memory store for the draft currently under review.

The tool is single-user and local by design, so the draft lives in process
memory and dies with the server. Persisting it is a later decision — the
workflow document puts storage at "local files initially, database later".
"""

from __future__ import annotations

import threading

from app.models.document import ParsedDocument
from app.models.survey import Question, QuestionDraft


class DraftStore:
    """Holds the one draft the survey programmer is working on."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._draft = QuestionDraft()
        self._document: ParsedDocument | None = None

    def get(self) -> QuestionDraft:
        with self._lock:
            return self._draft.model_copy(deep=True)

    def set_document(self, document: ParsedDocument) -> None:
        """Remember the parsed document so later batches need no re-upload.

        Replacing the document starts a new working set: a different
        questionnaire must not merge into the previous one's questions.
        """
        with self._lock:
            same_source = (
                self._document is not None
                and self._document.source_filename == document.source_filename
            )
            self._document = document.model_copy(deep=True)
            if not same_source:
                self._draft = QuestionDraft(source_filename=document.source_filename)
            else:
                self._draft.source_filename = document.source_filename

    def document(self) -> ParsedDocument | None:
        with self._lock:
            return None if self._document is None else self._document.model_copy(deep=True)

    def merge_questions(self, questions: list[Question]) -> None:
        """Add or replace questions by label, keeping everything else.

        Batching depends on this: classifying Q16-Q30 after Q1-Q15 must extend
        the working set, not replace it.
        """
        with self._lock:
            by_label = {q.label: q for q in self._draft.questions}
            for question in questions:
                by_label[question.label] = question.model_copy(deep=True)

            order = self._label_order()
            self._draft.questions = sorted(
                by_label.values(),
                key=lambda q: (order.get(q.label, len(order)), q.label),
            )

    def _label_order(self) -> dict[str, int]:
        """Document order, so a late batch still sorts into place."""
        if self._document is None:
            return {}
        return {
            boundary.label: position
            for position, boundary in enumerate(self._document.questions)
            if boundary.label
        }

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
            self._document = None


draft_store = DraftStore()
