"""The intermediate survey model — the contract between every phase.

The workflow document calls this "the heart of the product": it should stay
stable even if the XML syntax or the AI model changes. The AI populates it, the
survey programmer corrects it, and the XML generator consumes it. Nothing else
crosses between those stages.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.models.document import TextRun  # re-exported: one run type across all phases

__all__ = ["SUPPORTED_ELEMENTS", "TextRun", "OptionLine", "Question", "QuestionDraft"]

#: The Decipher elements V1 supports. Anything outside this list is a
#: programmer decision, not an AI one.
SUPPORTED_ELEMENTS: tuple[str, ...] = (
    "radio",
    "checkbox",
    "radio_grid",
    "checkbox_grid",
    "textarea",
    "text",
    "number",
    "select",
    "html",
)

#: Elements whose answer list lives in ``options``.
OPTION_ELEMENTS = frozenset({"radio", "checkbox", "select"})

#: Elements that carry both ``rows`` and ``cols``.
GRID_ELEMENTS = frozenset({"radio_grid", "checkbox_grid"})


class OptionLine(BaseModel):
    """One answer option, grid row, or grid column.

    Formatting is a line-level flag rather than a run list: options are short
    and the generator marks the whole line. Mixed formatting inside a single
    option collapses to "the line is bold if every part of it is bold".
    """

    raw_text: str
    bold: bool = False
    italic: bool = False

    @classmethod
    def from_runs(cls, runs: list[TextRun], text: str | None = None) -> OptionLine:
        meaningful = [run for run in runs if run.text.strip()]
        return cls(
            raw_text=text if text is not None else "".join(r.text for r in runs).strip(),
            bold=bool(meaningful) and all(run.bold for run in meaningful),
            italic=bool(meaningful) and all(run.italic for run in meaningful),
        )


class Question(BaseModel):
    """One classified question, ready for review and then generation."""

    label: str
    element: str
    title: list[TextRun] = Field(default_factory=list)
    comment: list[TextRun] = Field(default_factory=list)
    options: list[OptionLine] = Field(default_factory=list)
    rows: list[OptionLine] = Field(default_factory=list)
    cols: list[OptionLine] = Field(default_factory=list)
    confidence: float = 0.0
    needs_review: bool = True
    ai_notes: str = ""
    dev_notes: str = ""
    """Survey-programmer scratch notes. Never reaches the XML."""

    def title_text(self) -> str:
        return "".join(run.text for run in self.title).strip()

    def comment_text(self) -> str:
        return "".join(run.text for run in self.comment).strip()

    @property
    def is_grid(self) -> bool:
        return self.element in GRID_ELEMENTS


class QuestionDraft(BaseModel):
    """The full set of questions currently under review, plus its summary."""

    questions: list[Question] = Field(default_factory=list)
    source_filename: str | None = None
    review_threshold: float = 0.75

    @property
    def flagged_count(self) -> int:
        return sum(1 for question in self.questions if question.needs_review)

    @property
    def confident_count(self) -> int:
        return len(self.questions) - self.flagged_count

    def find(self, label: str) -> Question | None:
        return next((q for q in self.questions if q.label == label), None)

    def model_dump_with_summary(self) -> dict:
        payload = self.model_dump()
        payload["summary"] = {
            "total": len(self.questions),
            "flagged": self.flagged_count,
            "confident": self.confident_count,
        }
        return payload
