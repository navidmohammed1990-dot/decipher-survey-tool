"""Phase 2 endpoints — AI classification."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Body, Query
from pydantic import BaseModel, Field

from app.classify.classifier import classify_document
from app.classify.ollama import OllamaClient
from app.config import settings
from app.models.document import ParsedDocument
from app.models.survey import Question, QuestionDraft
from app.store import draft_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["classify"])


class ClassifyResponse(BaseModel):
    questions: list[Question] = Field(default_factory=list)
    source_filename: str | None = None
    review_threshold: float
    ai_available: bool
    """False when every question fell back to the heuristic."""
    fallback_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    summary: dict = Field(default_factory=dict)


def build_client() -> OllamaClient:
    return OllamaClient(
        base_url=settings.ollama_url,
        model=settings.ollama_model,
        timeout=settings.ollama_timeout,
    )


@router.post("/classify", response_model=ClassifyResponse)
def classify(
    document: ParsedDocument = Body(...),
    threshold: float = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence below this flags a question for review.",
    ),
) -> ClassifyResponse:
    """Classify a parsed document and store the result as the current draft.

    Defined with ``def`` rather than ``async def`` on purpose: the Ollama calls
    are blocking, so FastAPI runs this in a worker thread instead of stalling
    the event loop for the length of a whole questionnaire.
    """
    review_threshold = settings.review_threshold if threshold is None else threshold
    outcomes = classify_document(document, build_client(), review_threshold)

    questions = [outcome.question for outcome in outcomes]
    fallback_count = sum(1 for outcome in outcomes if outcome.used_fallback)

    warnings: list[str] = []
    if outcomes and fallback_count == len(outcomes):
        warnings.append(
            "The local AI was unreachable or returned unusable output for every "
            "question. All questions use the conservative fallback and need review."
        )
    elif fallback_count:
        warnings.append(f"{fallback_count} question(s) fell back to the heuristic.")

    draft = QuestionDraft(
        questions=questions,
        source_filename=document.source_filename,
        review_threshold=review_threshold,
    )
    draft_store.replace(draft)

    return ClassifyResponse(
        questions=questions,
        source_filename=document.source_filename,
        review_threshold=review_threshold,
        ai_available=bool(outcomes) and fallback_count < len(outcomes),
        fallback_count=fallback_count,
        warnings=warnings,
        summary={
            "total": len(questions),
            "flagged": draft.flagged_count,
            "confident": draft.confident_count,
        },
    )


@router.get("/ai-status", tags=["meta"])
def ai_status() -> dict:
    """Whether the local model is reachable, for the UI to surface up front."""
    client = build_client()
    return {
        "available": client.is_available(),
        "url": client.base_url,
        "model": client.model,
    }
