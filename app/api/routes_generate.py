"""Phase 3 endpoints — deterministic XML generation and export."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from app.generate.export import check_well_formed, wrap_survey
from app.generate.xml_generator import (
    UnsupportedElementError,
    generate_fragment,
    generate_questions,
)
from app.models.survey import Question
from app.store import draft_store

router = APIRouter(prefix="/api", tags=["generate"])


class GenerateRequest(BaseModel):
    questions: list[Question] | None = None
    """Omit to generate from the draft currently under review."""
    wrap: bool = False
    """Wrap in a placeholder ``<survey>`` root so the output parses standalone."""


class GenerateResponse(BaseModel):
    xml: str
    question_count: int
    well_formed: bool
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)


def _resolve(request: GenerateRequest) -> list[Question]:
    if request.questions is not None:
        return request.questions
    return draft_store.get().questions


@router.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    """Generate base Decipher XML.

    A pure function of its input: the same questions always produce the same
    bytes. No model is called here and nothing is stored.
    """
    questions = _resolve(request)
    if not questions:
        raise HTTPException(
            status_code=400,
            detail="No questions to generate. Classify a document first, or pass questions.",
        )

    try:
        fragments = generate_questions(questions)
    except UnsupportedElementError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result = check_well_formed(fragments)
    warnings = [
        f"{q.label} is still flagged for review." for q in questions if q.needs_review
    ]

    return GenerateResponse(
        xml=wrap_survey(fragments) if request.wrap else fragments,
        question_count=len(questions),
        well_formed=result.ok,
        error=result.error,
        warnings=warnings,
    )


@router.post("/generate/{label}", response_model=GenerateResponse)
def generate_one(label: str) -> GenerateResponse:
    """Preview a single question from the draft, for a sanity check pre-export."""
    question = draft_store.get().find(label)
    if question is None:
        raise HTTPException(status_code=404, detail=f"No question labelled '{label}'.")

    try:
        fragment = generate_fragment(question)
    except UnsupportedElementError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result = check_well_formed(fragment)
    return GenerateResponse(
        xml=fragment,
        question_count=1,
        well_formed=result.ok,
        error=result.error,
        warnings=[f"{label} is still flagged for review."] if question.needs_review else [],
    )


@router.get("/export.xml", response_class=PlainTextResponse)
def export_xml(
    wrap: bool = Query(True, description="Wrap in a placeholder <survey> root."),
) -> PlainTextResponse:
    """Download the assembled base XML for the current draft."""
    draft = draft_store.get()
    if not draft.questions:
        raise HTTPException(status_code=400, detail="No draft to export.")

    fragments = generate_questions(draft.questions)
    body = wrap_survey(fragments, name=draft.source_filename) if wrap else fragments

    stem = (draft.source_filename or "survey").rsplit(".", 1)[0]
    return PlainTextResponse(
        content=body,
        media_type="application/xml",
        headers={"Content-Disposition": f'attachment; filename="{stem}_base.xml"'},
    )
