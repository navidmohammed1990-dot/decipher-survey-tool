"""Phase 2 — the AI classifier.

The AI classifies. It never writes XML.

Its entire output is an element type plus which *line indices* are the title,
the comment, the options, and (grids only) the rows and columns. The text and
formatting that reach the generator are always Phase 1's. That separation is
what keeps a hallucinated word out of a live survey.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from app.classify.lines import SourceLine, question_lines
from app.classify.ollama import OllamaClient, OllamaError
from app.models.document import ParsedDocument, TextRun
from app.models.survey import (
    GRID_ELEMENTS,
    OPTION_ELEMENTS,
    SUPPORTED_ELEMENTS,
    OptionLine,
    Question,
)

logger = logging.getLogger(__name__)

DEFAULT_REVIEW_THRESHOLD = 0.75

FALLBACK_NOTE = "AI classification unavailable - please verify."

SYSTEM_PROMPT = """\
Classify ONE questionnaire question. Element must be one of: radio, checkbox,
radio_grid, checkbox_grid, textarea, text, number, select, html.
Identify which given line indices are the title, the comment/instruction,
the options, and (grids only) the rows and columns. Use ONLY given indices -
never invent or rewrite text. Respond with ONLY JSON:
{"element": "checkbox", "title_lines": [0], "comment_lines": [1],
 "option_lines": [2,3,4], "row_lines": [], "col_lines": [],
 "confidence": 0.9, "notes": "brief reason"}"""


class ClassificationOutcome(BaseModel):
    """One classified question plus how it was arrived at."""

    question: Question
    used_fallback: bool = False
    warnings: list[str] = Field(default_factory=list)


def build_prompt(label: str, lines: list[SourceLine]) -> str:
    body = "\n".join(line.prompt_form() for line in lines)
    return f"Question label: {label}\n\nLines:\n{body}\n"


# -- response interpretation ---------------------------------------------


def _index_list(payload: dict, key: str, valid: set[int]) -> tuple[list[int], list[str]]:
    """Read one index list from the model's JSON, discarding anything invalid."""
    raw = payload.get(key) or []
    if not isinstance(raw, list):
        return [], [f"'{key}' was not a list; ignored."]

    kept: list[int] = []
    dropped: list[int] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int):
            dropped.append(item)
        elif item not in valid:
            dropped.append(item)
        elif item not in kept:
            kept.append(item)

    warnings = [f"'{key}' referenced unknown lines {dropped}; ignored."] if dropped else []
    return kept, warnings


def _runs_for(indices: list[int], by_index: dict[int, SourceLine]) -> list[TextRun]:
    """Concatenate the runs of several lines into one run list."""
    runs: list[TextRun] = []
    for position, index in enumerate(indices):
        if position:
            runs.append(TextRun(text=" "))
        runs.extend(by_index[index].runs)
    return runs


def _options_for(indices: list[int], by_index: dict[int, SourceLine]) -> list[OptionLine]:
    return [
        OptionLine.from_runs(by_index[index].runs, by_index[index].text) for index in indices
    ]


def _coerce_confidence(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def interpret_response(
    payload: dict, label: str, lines: list[SourceLine], threshold: float
) -> ClassificationOutcome:
    """Map the model's index-only JSON back onto the parsed lines.

    Raises :class:`ValueError` when the response is unusable, so the caller can
    fall back rather than emitting a half-formed question.
    """
    by_index = {line.index: line for line in lines}
    valid = set(by_index)

    element = payload.get("element")
    if not isinstance(element, str) or element not in SUPPORTED_ELEMENTS:
        raise ValueError(f"Unsupported element {element!r}.")

    warnings: list[str] = []
    picks = {}
    for key in ("title_lines", "comment_lines", "option_lines", "row_lines", "col_lines"):
        picks[key], key_warnings = _index_list(payload, key, valid)
        warnings.extend(key_warnings)

    if not picks["title_lines"] and lines:
        picks["title_lines"] = [lines[0].index]
        warnings.append("No title line returned; defaulted to the first line.")

    confidence = _coerce_confidence(payload.get("confidence"))
    notes = payload.get("notes")
    question = Question(
        label=label,
        element=element,
        title=_runs_for(picks["title_lines"], by_index),
        comment=_runs_for(picks["comment_lines"], by_index),
        options=_options_for(picks["option_lines"], by_index),
        rows=_options_for(picks["row_lines"], by_index),
        cols=_options_for(picks["col_lines"], by_index),
        confidence=confidence,
        ai_notes=notes.strip() if isinstance(notes, str) else "",
    )

    warnings.extend(_structural_warnings(question))
    question.needs_review = confidence < threshold or bool(warnings)
    if warnings:
        question.ai_notes = " ".join(filter(None, [question.ai_notes, *warnings]))

    return ClassificationOutcome(question=question, warnings=warnings)


def _structural_warnings(question: Question) -> list[str]:
    """Catch classifications that cannot produce sensible XML.

    These do not discard the AI's answer — they force it in front of a human,
    which is the conservative half of the product principle.
    """
    warnings: list[str] = []
    if question.element in OPTION_ELEMENTS and not question.options:
        warnings.append(f"'{question.element}' has no options.")
    if question.element in GRID_ELEMENTS:
        if not question.rows:
            warnings.append(f"'{question.element}' has no rows.")
        if not question.cols:
            warnings.append(f"'{question.element}' has no columns.")
    if not question.title_text():
        warnings.append("Title is empty.")
    return warnings


def fallback_question(label: str, lines: list[SourceLine]) -> Question:
    """The conservative heuristic: title = first line, everything else options.

    Deliberately always low-confidence. An unreachable model must never look
    like a confident answer.
    """
    return Question(
        label=label,
        element="radio",
        title=list(lines[0].runs) if lines else [],
        options=_options_for([line.index for line in lines[1:]], {l.index: l for l in lines}),
        confidence=0.0,
        needs_review=True,
        ai_notes=FALLBACK_NOTE,
    )


# -- orchestration --------------------------------------------------------


def classify_question(
    label: str,
    lines: list[SourceLine],
    client: OllamaClient,
    threshold: float = DEFAULT_REVIEW_THRESHOLD,
) -> ClassificationOutcome:
    """Classify one question, degrading to the heuristic on any failure."""
    if not lines:
        return ClassificationOutcome(
            question=Question(
                label=label, element="html", confidence=0.0, needs_review=True,
                ai_notes="No content found for this question.",
            ),
            used_fallback=True,
            warnings=["No content found for this question."],
        )

    try:
        payload = client.generate_json(SYSTEM_PROMPT, build_prompt(label, lines))
        return interpret_response(payload, label, lines, threshold)
    except (OllamaError, ValueError) as exc:
        logger.warning("Falling back to heuristic for %s: %s", label, exc)
        return ClassificationOutcome(
            question=fallback_question(label, lines),
            used_fallback=True,
            warnings=[str(exc)],
        )


def classify_document(
    document: ParsedDocument,
    client: OllamaClient,
    threshold: float = DEFAULT_REVIEW_THRESHOLD,
) -> list[ClassificationOutcome]:
    """Classify every non-preamble question, one model call each.

    One call per question rather than one per document: the brief is explicit
    that this gives better accuracy and avoids context-length dropouts.
    """
    outcomes: list[ClassificationOutcome] = []
    for boundary in document.questions:
        if boundary.is_preamble or not boundary.label:
            continue
        lines = question_lines(document, boundary)
        outcomes.append(classify_question(boundary.label, lines, client, threshold))
    return outcomes
