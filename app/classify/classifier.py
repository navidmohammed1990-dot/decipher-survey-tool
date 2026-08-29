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

from app.classify.corrections import correction_memory
from app.classify.seed_library import prompt_prefix as seed_prefix
from app.classify.lines import SourceLine, marker_code, question_lines
from app.classify.ollama import OllamaClient, OllamaError
from app.models.document import ParsedDocument, TextRun
from app.generate.resources import DEFAULT_SUBJECT_TYPE, SUBJECT_TYPES, resource_tag_for
from app.models.survey import (
    EXCLUDED_ELEMENTS,
    NO_XML_ELEMENTS,
    NON_QUESTION_ELEMENTS,
    ClassificationTrace,
    GRID_ELEMENTS,
    OPTION_ELEMENTS,
    SUPPORTED_ELEMENTS,
    OptionLine,
    Question,
)

logger = logging.getLogger(__name__)

DEFAULT_REVIEW_THRESHOLD = 0.75

#: Every line-role key the model may return.
ROLE_KEYS = (
    "title_lines",
    "comment_lines",
    "option_lines",
    "routing_lines",
    "type_signal_lines",
    "row_lines",
    "col_lines",
)

#: Which element a house type marker implies, when options are present.
TYPE_TAG_ELEMENTS = {
    "SC": {"radio", "select"},
    "SR": {"radio", "select"},
    "MC": {"checkbox"},
    "MR": {"checkbox"},
    "SR_GRID": {"radio_grid"},
    "MR_GRID": {"checkbox_grid"},
    "OE": {"textarea", "text"},
    "NUM": {"number"},
    "GRID": {"radio_grid", "checkbox_grid"},
}

FALLBACK_NOTE = "AI classification unavailable - please verify."

SYSTEM_PROMPT = """\
You are a survey programmer's assistant, classifying every line of ONE
questionnaire question. For each line you're given the raw text plus some
factual hints detected about its formatting - use these as evidence, the
way an experienced programmer would notice formatting cues, but make the
actual judgment call yourself. Hints are not rules - a line can be red
without being an instruction, or plain text and still be an instruction.
Decide based on what the line actually says and how it's formatted together.

Classify each line's role:
- title: the question being asked
- comment: an instruction to the respondent (e.g. "select all that apply")
- option: a response option the respondent could select
- routing: a note to the programmer (skip logic, quotas, termination,
  randomization instructions) - never shown to respondents, never an option
- type_signal: an explicit type marker like "ASK ALL, SC"

A type marker or "ASK ..." header that appears AFTER this question's options
usually introduces the NEXT question - classify it as routing and do not use
it as this question's type signal.

Then decide the element type: radio, checkbox, radio_grid, checkbox_grid,
textarea, text, number, select, html, not_a_question, custom_complex.

Choose custom_complex for a real respondent task no standard element can
express - a gamified image quiz, a slider synced to video playback, anything
needing bespoke scripting. Saying so is more useful than a wrong approximation.

Choose not_a_question when the WHOLE block speaks to the programmer, not the
respondent - a derived-variable definition, a coding instruction. Evidence,
not rules: bracketed instruction phrasing ("Please create...", "auto code
based on..."); a reference to an already-asked question ("based on S2 Age");
a variable-name token like S2_AGE BANDS instead of a sentence; no
natural-language question anywhere. If the block does ask the respondent
something, it is a question - put programmer notes in routing_lines instead.

Use the type_signal if one exists and
options are present (SC->radio, MC->checkbox) unless the options clearly
form a grid (row statements x column scale). If no options exist, ignore
any type_signal and classify from the question's wording instead.

A grid asks the SAME scale about SEVERAL things. Judge that from meaning, not
a remembered phrase: "SR PER ROW", "SR per statement", "MR per brand", "one
answer for each item", "rate each of the following", or nothing at all. Layout
does not matter either - the scale may be pipe, tab or space separated, or
listed above the statements. If one set of lines is a repeated answer scale and
another is the things being rated, it is a grid: scale in col_lines, things in
row_lines, option_lines empty. SR/single -> radio_grid, MR/multi ->
checkbox_grid.

For grids only, also say what the ROWS describe with subject_type: one of
brand, category, product, statement, none. Use "statement" if unclear.

Never invent or rewrite text - only classify given lines by index.

Respond with ONLY this JSON:
{"element": "checkbox", "title_lines": [0], "comment_lines": [1],
 "option_lines": [2,3,4], "routing_lines": [5], "type_signal_lines": [],
 "row_lines": [], "col_lines": [], "subject_type": "none",
 "confidence": 0.9, "notes": "brief reasoning, including how you used any hints"}"""


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
    # Document order, not the order the model happened to list them in. A model
    # that reasons its way to the main clause first can return title_lines as
    # [1, 0], which would otherwise assemble the title backwards.
    return sorted(kept), warnings


def _runs_for(indices: list[int], by_index: dict[int, SourceLine]) -> list[TextRun]:
    """Concatenate the runs of several lines into one run list."""
    runs: list[TextRun] = []
    for position, index in enumerate(indices):
        if position:
            runs.append(TextRun(text=" "))
        runs.extend(by_index[index].runs)
    return runs


def _options_for(indices: list[int], by_index: dict[int, SourceLine]) -> list[OptionLine]:
    """Build options, preserving whatever code the source gave each one.

    A code can arrive trailing ("Other | 97") or as a typed leading number
    ("97. Other"). Both are the author's choice and outrank renumbering.
    """
    options: list[OptionLine] = []
    for index in indices:
        line = by_index[index]
        option = OptionLine.from_runs(line.runs, line.text)
        if option.code is None:
            option.code = marker_code(line.literal_marker)
        options.append(option)
    return options


def _coerce_confidence(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0



def _coerce_subject_type(value, element: str) -> str:
    """What a grid's rows describe.

    Only grids carry a subject; for anything else it is always "none". An
    unusable answer on a grid becomes "statement", the most generic wording.
    """
    if element not in GRID_ELEMENTS:
        return "none"
    if isinstance(value, str) and value.lower() in SUBJECT_TYPES:
        resolved = value.lower()
        return DEFAULT_SUBJECT_TYPE if resolved == "none" else resolved
    return DEFAULT_SUBJECT_TYPE



def _non_question_outcome(
    label: str, element: str, lines: list[SourceLine], confidence: float, notes: str
) -> ClassificationOutcome:
    """A block the model judged to be programmer content, not a question.

    Every line is preserved for the programmer to read and copy, and none of it
    is put through option numbering or any other question-shaped processing.
    """
    return ClassificationOutcome(
        question=Question(
            label=label,
            element=element,
            confidence=confidence,
            # Not flagged: this is a definite answer, not an uncertain one.
            needs_review=False,
            ai_notes=notes,
            routing_notes=[line.text for line in lines],
            trace=ClassificationTrace(lines=[line.text for line in lines]),
        )
    )


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
    for key in ROLE_KEYS:
        picks[key], key_warnings = _index_list(payload, key, valid)
        warnings.extend(key_warnings)

    if not picks["title_lines"] and lines:
        picks["title_lines"] = [lines[0].index]
        warnings.append("No title line returned; defaulted to the first line.")

    confidence = _coerce_confidence(payload.get("confidence"))
    notes = payload.get("notes")
    notes_text = notes.strip() if isinstance(notes, str) else ""
    subject_type = _coerce_subject_type(payload.get("subject_type"), element)

    # Programmer-facing lines are never respondent content and never reach XML.
    routing_notes = [
        by_index[index].text
        for index in picks["routing_lines"] + picks["type_signal_lines"]
    ]

    if element in NO_XML_ELEMENTS:
        # The whole block is programmer content. Keep every line verbatim for
        # reference and build nothing: no title, no options, no row numbering.
        return _non_question_outcome(label, element, lines, confidence, notes_text)
    question = Question(
        label=label,
        element=element,
        subject_type=subject_type,
        comment_resource=resource_tag_for(element, subject_type),
        title=_runs_for(picks["title_lines"], by_index),
        comment=_runs_for(picks["comment_lines"], by_index),
        options=_options_for(picks["option_lines"], by_index),
        rows=_options_for(picks["row_lines"], by_index),
        cols=_options_for(picks["col_lines"], by_index),
        confidence=confidence,
        ai_notes=notes_text,
        routing_notes=routing_notes,
        trace=ClassificationTrace(
            lines=[line.text for line in lines],
            ai_payload={key: payload.get(key) for key in ("element", *ROLE_KEYS, "subject_type")},
        ),
    )

    warnings.extend(_structural_warnings(question))
    warnings.extend(_dropped_option_warnings(question, lines, picks))
    warnings.extend(disagreements(question, payload, lines, picks, notes_text))
    question.needs_review = confidence < threshold or bool(warnings)
    if warnings:
        question.ai_notes = " ".join(filter(None, [question.ai_notes, *warnings]))

    return ClassificationOutcome(question=question, warnings=warnings)



def disagreements(
    question: Question,
    payload: dict,
    lines: list[SourceLine],
    picks: dict[str, list[int]],
    notes: str,
) -> list[str]:
    """Where strong pattern evidence contradicts the model, say so.

    This is a safety net, not a gate: nothing is auto-corrected, because
    picking a side silently is exactly what puts a wrong answer into a survey.
    The question is flagged and the programmer decides.
    """
    by_index = {line.index: line for line in lines}
    found: list[str] = []

    for index in picks["option_lines"] + picks["row_lines"] + picks["col_lines"]:
        line = by_index[index]
        if line.features.matches_routing_keyword:
            found.append(
                f"Line {index} ({line.text[:60]!r}) looks like routing text by keyword "
                f"but the model classified it as an option - please check."
            )

    # The reverse: a line the hints say nothing about, called routing with no
    # reasoning offered. If the model explained itself, trust its judgment —
    # house conventions the hints do not cover are exactly what it is for.
    if not notes:
        for index in picks["routing_lines"]:
            line = by_index[index]
            features = line.features
            if not (features.matches_routing_keyword or features.matches_type_tag_pattern
                    or features.is_colored):
                found.append(
                    f"Line {index} ({line.text[:60]!r}) was classified as routing but "
                    f"carries no routing hint and the model gave no reasoning - please check."
                )

    found.extend(_type_tag_disagreement(question, lines, picks))
    return found


def _type_tag_disagreement(
    question: Question, lines: list[SourceLine], picks: dict[str, list[int]]
) -> list[str]:
    """Flag an element that contradicts an explicit SC/MC/OE marker."""
    if question.element in GRID_ELEMENTS:
        # The prompt allows a grid to override the marker.
        return []
    if not (question.options or question.rows):
        # With no options the marker says nothing about the element.
        return []

    content = picks["option_lines"] + picks["row_lines"]
    first_content = min(content) if content else None

    for line in lines:
        tag = line.features.type_tag_value
        if not tag or tag not in TYPE_TAG_ELEMENTS:
            continue
        if first_content is not None and line.index > first_content:
            # A marker after the options belongs to the next question.
            continue
        expected = TYPE_TAG_ELEMENTS[tag]
        if question.element not in expected:
            return [
                f"Line {line.index} marks this question as '{tag}' but the model chose "
                f"'{question.element}' (expected {' or '.join(sorted(expected))}) - please check."
            ]
    return []



#: An unclaimed line only counts as a missed option if it looks like one.
_MAX_OPTION_CHARS = 90

#: Below this many unclaimed option-shaped lines, silence is unremarkable.
_MIN_UNCLAIMED_TO_FLAG = 3


def _option_shaped(line: SourceLine) -> bool:
    """Whether an unassigned line plausibly belongs in the option list."""
    text = line.text.strip()
    if not text or len(text) > _MAX_OPTION_CHARS or text.endswith("?"):
        return False
    return not line.features.matches_routing_keyword


def _dropped_option_warnings(
    question: Question, lines: list[SourceLine], picks: dict[str, list[int]]
) -> list[str]:
    """Catch an option list that quietly lost most of its options.

    An eight-band list that produced one row exported a near-empty question
    without complaint. Rather than guess which lines were meant, say the
    numbers do not add up and let the programmer look.
    """
    if question.element not in OPTION_ELEMENTS | GRID_ELEMENTS:
        return []

    chosen = sum(len(picks[key]) for key in ROLE_KEYS)
    if not chosen:
        return []

    claimed = {index for key in ROLE_KEYS for index in picks[key]}
    unclaimed = [line for line in lines if line.index not in claimed and _option_shaped(line)]
    selected = len(question.options) + len(question.rows) + len(question.cols)

    if len(unclaimed) >= _MIN_UNCLAIMED_TO_FLAG and len(unclaimed) > selected:
        return [
            f"Only {selected} option(s) were taken from this block, but "
            f"{len(unclaimed)} more lines look like options and were left out "
            f"- please check nothing was dropped."
        ]
    return []

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
        comment_resource=resource_tag_for("radio"),
        title=list(lines[0].runs) if lines else [],
        options=_options_for([line.index for line in lines[1:]], {l.index: l for l in lines}),
        confidence=0.0,
        needs_review=True,
        ai_notes=FALLBACK_NOTE,
        trace=ClassificationTrace(lines=[line.text for line in lines]),
    )


# -- orchestration --------------------------------------------------------


def classify_question(
    label: str,
    lines: list[SourceLine],
    client: OllamaClient,
    threshold: float = DEFAULT_REVIEW_THRESHOLD,
    system_prefix: str | None = None,
) -> ClassificationOutcome:
    """Classify one question, degrading to the heuristic on any failure.

    ``system_prefix`` defaults to the corrections recorded for the document
    under review. Pass ``""`` to opt out — Quick Convert does, so one
    questionnaire's house conventions cannot leak into an unrelated paste.
    """
    if not lines:
        return ClassificationOutcome(
            question=Question(
                label=label, element="html", confidence=0.0, needs_review=True,
                ai_notes="No content found for this question.",
            ),
            used_fallback=True,
            warnings=["No content found for this question."],
        )

    if lines and all(line.features.is_struck for line in lines):
        # Struck content is deleted content. That is formatting, not a judgment
        # call, so it needs no model call — and skipping one is time saved.
        return ClassificationOutcome(
            question=Question(
                label=label,
                element="excluded",
                confidence=1.0,
                needs_review=False,
                ai_notes="Struck through in the source, so treated as deleted "
                         "and not converted.",
                routing_notes=[line.text for line in lines],
                trace=ClassificationTrace(lines=[line.text for line in lines]),
            )
        )

    try:
        # Seeded examples are always carried; document corrections are not,
        # since they belong to the document currently under review.
        prefix = correction_memory.prompt_prefix() if system_prefix is None else system_prefix
        system = seed_prefix() + prefix + SYSTEM_PROMPT
        payload = client.generate_json(system, build_prompt(label, lines))
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
