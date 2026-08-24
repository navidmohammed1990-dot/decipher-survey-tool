"""Phase 2 endpoints — AI classification."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel, Field

from app.classify.classifier import classify_document
from app.classify.jobs import job_manager
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

    draft_store.set_document(document)
    draft_store.merge_questions(questions)
    draft = draft_store.get()
    draft.review_threshold = review_threshold
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


# -- batched, backgrounded classification ---------------------------------


class StartJobRequest(BaseModel):
    document: ParsedDocument
    labels: list[str] | None = None
    """Question labels to classify in this batch. Omit for all of them."""
    threshold: float | None = None


class StartJobResponse(BaseModel):
    job_id: str
    total: int
    labels: list[str]
    slow_warning_seconds: float


def _available_labels(document: ParsedDocument) -> list[str]:
    return [b.label for b in document.questions if b.label and not b.is_preamble]


@router.post("/classify/start", response_model=StartJobResponse)
def start_classification(request: StartJobRequest) -> StartJobResponse:
    """Begin classifying a batch in the background.

    Parsing already covered the whole document, so the programmer can pick a
    subset here rather than waiting on every question at once.
    """
    available = _available_labels(request.document)
    if not available:
        raise HTTPException(status_code=400, detail="This document has no questions to classify.")

    if request.labels is None:
        labels = available
    else:
        known = set(available)
        labels = [label for label in request.labels if label in known]
        unknown = [label for label in request.labels if label not in known]
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown question label(s): {', '.join(unknown)}.",
            )
        if not labels:
            raise HTTPException(status_code=400, detail="No questions selected.")

    threshold = settings.review_threshold if request.threshold is None else request.threshold

    # Register the document first so results merge into one working set, and so
    # a later batch needs no re-upload.
    draft_store.set_document(request.document)
    draft = draft_store.get()
    draft.review_threshold = threshold
    draft_store.replace(draft)

    job = job_manager.start(
        document=request.document,
        labels=labels,
        client=build_client(),
        threshold=threshold,
        # Merge as each question lands, so cancelling keeps completed work.
        on_batch=draft_store.merge_questions,
    )
    return StartJobResponse(
        job_id=job.id,
        total=job.total,
        labels=labels,
        slow_warning_seconds=settings.slow_classify_seconds,
    )


@router.get("/classify/{job_id}/status")
def classification_status(job_id: str) -> dict:
    """Poll a running batch. Drives the progress bar and the slow warning."""
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No classification job '{job_id}'.")

    status = job.status()
    remaining = status["estimated_remaining_seconds"]
    status["slow_warning_seconds"] = settings.slow_classify_seconds
    status["slow"] = remaining is not None and remaining > settings.slow_classify_seconds
    return status


@router.post("/classify/{job_id}/cancel")
def cancel_classification(job_id: str) -> dict:
    """Stop a batch without discarding what it already finished."""
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No classification job '{job_id}'.")

    job.cancel()
    return job.status()
