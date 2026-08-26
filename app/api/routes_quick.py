"""Phase 8 — Quick Convert: paste text, get XML.

A second entry point, not a replacement. Whole-document parsing is where the
hard problems live; a pasted chunk skips them because the programmer already
did the segmentation by choosing what to select.

Everything below the input is reused unchanged: the Phase 7 classifier judges
the lines, the Phase 6 generator writes the XML. This module owns only the
input path.

Deliberately stateless. It never touches the draft store or the correction
memory, so a Quick Convert never disturbs a DOCX review in progress.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel, Field

from app.api import routes_classify
from app.classify.classifier import classify_question
from app.classify.paste import split_questions
from app.config import settings
from app.generate.export import check_well_formed
from app.generate.xml_generator import UnsupportedElementError, generate_questions
from app.models.survey import Question

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["quick"])

#: Quick Convert is for chunks the programmer selected, typically 1-5
#: questions. A larger paste is a signal to paste less, not for the tool to
#: start managing batches.
MAX_PASTE_CHARS = 60_000


class QuickConvertRequest(BaseModel):
    text: str
    threshold: float | None = None


class QuickConvertResponse(BaseModel):
    questions: list[Question] = Field(default_factory=list)
    xml: str = ""
    well_formed: bool = True
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    ai_available: bool = True
    ai_detail: str = ""
    fallback_count: int = 0
    timings: list[dict] = Field(default_factory=list)
    """What Ollama reported for each call: load, prompt and generation time.

    Surfaced because generation speed is the evidence that separates GPU from
    CPU inference, and it is otherwise invisible from inside the app.
    """


@router.post("/quick-convert", response_model=QuickConvertResponse)
def quick_convert(
    request: QuickConvertRequest = Body(...),
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
) -> QuickConvertResponse:
    """Split a paste, classify each question, and generate the XML.

    Synchronous on purpose: chunks are small, so a spinner is enough and the
    progress/ETA machinery the DOCX path needs would only add latency here.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Nothing to convert - paste some text first.")
    if len(request.text) > MAX_PASTE_CHARS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"That paste is {len(request.text):,} characters. Quick Convert is for "
                f"chunks of a few questions - paste a smaller selection, or use the "
                f"document upload for a whole questionnaire."
            ),
        )

    blocks, warnings = split_questions(request.text)
    if not blocks:
        raise HTTPException(status_code=422, detail=" ".join(warnings) or "No questions found.")

    review_threshold = (
        settings.review_threshold
        if (threshold if threshold is not None else request.threshold) is None
        else (threshold if threshold is not None else request.threshold)
    )

    # Called through the module so the client factory stays a single
    # override point for both entry points.
    client = routes_classify.build_client()
    ai_status = client.status()

    questions: list[Question] = []
    fallback_count = 0
    for block in blocks:
        outcome = classify_question(
            block.label,
            block.lines,
            client,
            review_threshold,
            # Opt out of the DOCX session's corrections: they belong to that
            # document, not to an unrelated paste.
            system_prefix="",
        )
        questions.append(outcome.question)
        fallback_count += 1 if outcome.used_fallback else 0

    if fallback_count and not ai_status["available"]:
        warnings.append(f"{ai_status['detail']} Classified with the conservative fallback.")
    elif fallback_count:
        warnings.append(f"{fallback_count} question(s) fell back to the heuristic.")

    xml, well_formed, error = _render(questions)
    return QuickConvertResponse(
        questions=questions,
        timings=list(client.stats),
        xml=xml,
        well_formed=well_formed,
        error=error,
        warnings=warnings,
        ai_available=ai_status["available"],
        ai_detail=ai_status["detail"],
        fallback_count=fallback_count,
    )


class QuickGenerateRequest(BaseModel):
    questions: list[Question] = Field(default_factory=list)


@router.post("/quick-generate", response_model=QuickConvertResponse)
def quick_generate(request: QuickGenerateRequest) -> QuickConvertResponse:
    """Re-generate from edited questions, without re-classifying.

    Editing a field and seeing the XML update must not cost another model call.
    """
    if not request.questions:
        raise HTTPException(status_code=400, detail="No questions to generate.")

    xml, well_formed, error = _render(request.questions)
    return QuickConvertResponse(
        questions=request.questions, xml=xml, well_formed=well_formed, error=error
    )


def _render(questions: list[Question]) -> tuple[str, bool, str | None]:
    try:
        xml = generate_questions(questions)
    except UnsupportedElementError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result = check_well_formed(xml)
    return xml, result.ok, result.error
