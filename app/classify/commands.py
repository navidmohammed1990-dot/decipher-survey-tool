"""Interpreting a survey programmer's plain-English correction.

Strictly additive. The dropdowns and text fields stay the primary, instant way
to correct anything; this costs an extra model round trip, so it is offered as
a convenience and never as the only route.

The same discipline as the main classifier applies: structured JSON out, no
XML, and nothing invented that the instruction did not ask for. An instruction
the model cannot map cleanly says so rather than guessing — a silently wrong
edit is worse than no edit.
"""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel, Field

from app.classify.ollama import OllamaClient, OllamaError
from app.models.survey import SUPPORTED_ELEMENTS, OptionLine, Question
from app.models.document import TextRun

logger = logging.getLogger(__name__)

#: Fields an instruction may change. Everything else is off limits.
EDITABLE_FIELDS = ("element", "title", "comment", "options", "rows", "cols", "routing_notes")

UNCLEAR_MESSAGE = "Not sure what you meant, please use the fields below."

SYSTEM_PROMPT = """\
You turn a survey programmer's plain-English correction into a structured edit
of ONE question. You are not classifying the question from scratch - you are
applying only what the instruction asks for.

Rules:
- Change only the fields the instruction actually mentions. Leave everything
  else out of your answer entirely; omitted fields keep their current value.
- Never invent option text, titles or codes the instruction does not give. If
  the instruction refers to options "shown below" or "as listed", it means the
  options the question already has - do not restate them.
- element must be one of: {elements}.
- If the instruction is ambiguous, contradictory, or not about editing this
  question, set understood to false and say why. Do not guess.

Respond with ONLY JSON:
{{"understood": true, "changes": {{"element": "radio"}}, "reason": "brief"}}
or
{{"understood": false, "changes": {{}}, "reason": "why you could not apply it"}}

Text fields (title, comment) are plain strings. List fields (options, rows,
cols, routing_notes) are arrays of strings, one entry per line."""


class CommandResult(BaseModel):
    """What an instruction did, ready for the programmer to confirm."""

    understood: bool = False
    reason: str = ""
    question: Question | None = None
    """The edited question, not yet saved. ``None`` when nothing was applied."""
    changes: list[dict] = Field(default_factory=list)
    """Per-field before/after, so a misread instruction is visible."""


def build_prompt(question: Question, instruction: str) -> str:
    """Show the model the question as it stands, then the instruction."""
    current = {
        "label": question.label,
        "element": question.element,
        "title": question.title_text(),
        "comment": question.comment_text(),
        "options": [o.raw_text for o in question.options],
        "rows": [r.raw_text for r in question.rows],
        "cols": [c.raw_text for c in question.cols],
        "routing_notes": list(question.routing_notes),
    }
    return (
        f"Current question:\n{json.dumps(current, indent=2)}\n\n"
        f"Programmer's instruction:\n{instruction.strip()}\n"
    )


def _as_lines(value) -> list[str] | None:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    return None


def _describe(value) -> str:
    if isinstance(value, list):
        return "\n".join(value)
    return str(value)


def apply_changes(question: Question, changes: dict) -> tuple[Question, list[dict]]:
    """Apply an interpreted edit, returning the copy and what it altered."""
    updated = question.model_copy(deep=True)
    applied: list[dict] = []

    def note(field: str, before, after) -> None:
        if _describe(before) != _describe(after):
            applied.append({"field": field, "before": _describe(before), "after": _describe(after)})

    for field, value in changes.items():
        if field not in EDITABLE_FIELDS or value is None:
            continue

        if field == "element":
            if not isinstance(value, str) or value not in SUPPORTED_ELEMENTS:
                continue
            note("element", updated.element, value)
            updated.element = value

        elif field in ("title", "comment"):
            text = value if isinstance(value, str) else None
            if text is None:
                continue
            before = updated.title_text() if field == "title" else updated.comment_text()
            note(field, before, text.strip())
            runs = [TextRun(text=text.strip())] if text.strip() else []
            setattr(updated, field, runs)
            if field == "comment" and text.strip():
                # An explicit comment means custom text, not a resource tag.
                updated.comment_resource = None

        elif field == "routing_notes":
            lines = _as_lines(value)
            if lines is None:
                continue
            note(field, updated.routing_notes, lines)
            updated.routing_notes = lines

        else:  # options, rows, cols
            lines = _as_lines(value)
            if lines is None:
                continue
            before = [o.raw_text for o in getattr(updated, field)]
            note(field, before, lines)
            setattr(updated, field, [OptionLine.from_text(line) for line in lines])

    return updated, applied


def interpret(
    question: Question, instruction: str, client: OllamaClient
) -> CommandResult:
    """Turn one instruction into a proposed edit, or explain why it cannot."""
    if not instruction.strip():
        return CommandResult(understood=False, reason="No instruction was given.")

    system = SYSTEM_PROMPT.format(elements=", ".join(SUPPORTED_ELEMENTS))
    try:
        payload = client.generate_json(system, build_prompt(question, instruction))
    except OllamaError as exc:
        logger.warning("Correction command failed for %s: %s", question.label, exc)
        return CommandResult(understood=False, reason=str(exc))

    reason = payload.get("reason")
    reason = reason.strip() if isinstance(reason, str) else ""

    if not payload.get("understood"):
        return CommandResult(understood=False, reason=reason or UNCLEAR_MESSAGE)

    changes = payload.get("changes")
    if not isinstance(changes, dict) or not changes:
        return CommandResult(understood=False, reason=reason or UNCLEAR_MESSAGE)

    updated, applied = apply_changes(question, changes)
    if not applied:
        # Understood, but it asked for nothing this question does not already
        # have. Say that plainly — the model's own wording ("already") does not
        # tell the programmer why no confirmation dialog appeared.
        detail = f" ({reason})" if reason else ""
        return CommandResult(
            understood=False,
            reason=f"That instruction would not change anything on this question.{detail}",
        )

    return CommandResult(understood=True, reason=reason, question=updated, changes=applied)
