"""Phase 4 endpoints — the survey programmer's review flow.

Every field is editable regardless of the AI's confidence. A high score is a
hint about where to look first, never a lock: the programmer controls the
final decision.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.models.survey import SUPPORTED_ELEMENTS, OptionLine, Question
from app.store import draft_store

router = APIRouter(prefix="/api", tags=["review"])


class DraftResponse(BaseModel):
    questions: list[Question] = Field(default_factory=list)
    source_filename: str | None = None
    review_threshold: float = 0.75
    summary: dict = Field(default_factory=dict)


def _draft_response() -> DraftResponse:
    draft = draft_store.get()
    return DraftResponse(
        questions=draft.questions,
        source_filename=draft.source_filename,
        review_threshold=draft.review_threshold,
        summary={
            "total": len(draft.questions),
            "flagged": draft.flagged_count,
            "confident": draft.confident_count,
        },
    )


class QuestionPatch(BaseModel):
    """A partial edit. Every field is optional; only what is sent changes.

    Options, rows and columns arrive as line-separated text because that is how
    a programmer edits them in a textarea — one option per line.
    """

    element: str | None = None
    title: str | None = None
    comment: str | None = None
    options: str | None = None
    rows: str | None = None
    cols: str | None = None
    dev_notes: str | None = None
    needs_review: bool | None = None

    @field_validator("element")
    @classmethod
    def _known_element(cls, value: str | None) -> str | None:
        if value is not None and value not in SUPPORTED_ELEMENTS:
            raise ValueError(
                f"'{value}' is not supported. Expected one of: {', '.join(SUPPORTED_ELEMENTS)}."
            )
        return value


def _lines_to_options(text: str) -> list[OptionLine]:
    return [
        OptionLine(raw_text=line.strip())
        for line in text.splitlines()
        if line.strip()
    ]


def _text_to_runs(text: str):
    """Programmer-typed text is plain: one unformatted run, or none if empty.

    Editing a title by hand deliberately drops the imported bold/italic for
    that field — the programmer is replacing it, and silently reapplying the
    old formatting to new words would be worse than losing it.
    """
    from app.models.document import TextRun

    return [TextRun(text=text)] if text.strip() else []


def apply_patch(question: Question, patch: QuestionPatch) -> Question:
    """Return a copy of ``question`` with the patch applied."""
    updated = question.model_copy(deep=True)

    if patch.element is not None:
        updated.element = patch.element
    if patch.title is not None:
        updated.title = _text_to_runs(patch.title)
    if patch.comment is not None:
        updated.comment = _text_to_runs(patch.comment)
    if patch.options is not None:
        updated.options = _lines_to_options(patch.options)
    if patch.rows is not None:
        updated.rows = _lines_to_options(patch.rows)
    if patch.cols is not None:
        updated.cols = _lines_to_options(patch.cols)
    if patch.dev_notes is not None:
        updated.dev_notes = patch.dev_notes

    if patch.needs_review is not None:
        updated.needs_review = patch.needs_review
    elif _changes_structure(patch):
        # A programmer edit is a decision, so clear the flag that asked for one.
        updated.needs_review = False

    return updated


def _changes_structure(patch: QuestionPatch) -> bool:
    return any(
        value is not None
        for value in (patch.element, patch.title, patch.comment,
                      patch.options, patch.rows, patch.cols)
    )


@router.get("/questions", response_model=DraftResponse)
def list_questions() -> DraftResponse:
    """The draft currently under review."""
    return _draft_response()


@router.patch("/questions/{label}", response_model=Question)
def patch_question(label: str, patch: QuestionPatch) -> Question:
    """Edit any field of one question. Nothing is locked by AI confidence."""
    question = draft_store.get().find(label)
    if question is None:
        raise HTTPException(status_code=404, detail=f"No question labelled '{label}'.")

    updated = draft_store.update_question(label, apply_patch(question, patch))
    if updated is None:  # pragma: no cover - lost a race with a clear()
        raise HTTPException(status_code=404, detail=f"No question labelled '{label}'.")
    return updated


@router.delete("/questions", status_code=204)
def clear_draft() -> None:
    """Discard the current draft."""
    draft_store.clear()
