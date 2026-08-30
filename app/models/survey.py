"""The intermediate survey model — the contract between every phase.

The workflow document calls this "the heart of the product": it should stay
stable even if the XML syntax or the AI model changes. The AI populates it, the
survey programmer corrects it, and the XML generator consumes it. Nothing else
crosses between those stages.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.models.document import TextRun  # re-exported: one run type across all phases

__all__ = [
    "SUPPORTED_ELEMENTS",
    "NON_QUESTION_ELEMENTS",
    "CUSTOM_ELEMENTS",
    "EXCLUDED_ELEMENTS",
    "NO_XML_ELEMENTS",
    "SubjectType",
    "TextRun",
    "OptionLine",
    "ClassificationTrace",
    "Question",
    "QuestionDraft",
]

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
    "not_a_question",
    "custom_complex",
    "excluded",
)

#: Elements that describe no respondent-facing question at all. A programmer
#: instruction or a derived-variable definition is meta-content like a routing
#: note, except that it is the *whole* block rather than one line inside a
#: real question. Nothing here is ever given a question shape.
NON_QUESTION_ELEMENTS = frozenset({"not_a_question"})

#: A real question the tool cannot express: a gamified task, a slider synced to
#: video. Recognising it and saying so beats emitting a wrong approximation.
CUSTOM_ELEMENTS = frozenset({"custom_complex"})

#: Content struck through in the source. Deleted, so never converted.
EXCLUDED_ELEMENTS = frozenset({"excluded"})

#: Everything that yields no XML, for whatever reason. The generator cares only
#: that nothing is emitted; the distinction matters to the person reading it.
NO_XML_ELEMENTS = NON_QUESTION_ELEMENTS | CUSTOM_ELEMENTS | EXCLUDED_ELEMENTS

#: Elements whose answer list lives in ``options``.
OPTION_ELEMENTS = frozenset({"radio", "checkbox", "select"})

#: Elements that carry both ``rows`` and ``cols``.
GRID_ELEMENTS = frozenset({"radio_grid", "checkbox_grid"})

#: What a grid's rows describe. Only meaningful for grids; it selects between
#: the ``SRBrand`` / ``SRCategory`` / ``SRProduct`` / ``SRStatement`` variants.
SubjectType = Literal["brand", "category", "product", "statement", "none"]


class OptionLine(BaseModel):
    """One answer option, grid row, or grid column.

    Formatting is a line-level flag rather than a run list: options are short
    and the generator marks the whole line. Mixed formatting inside a single
    option collapses to "the line is bold if every part of it is bold".
    """

    raw_text: str
    bold: bool = False
    italic: bool = False
    code: str | None = None
    """A value the source document gave this option, e.g. the 97 in ``Other | 97``.

    Preserved verbatim rather than renumbered: 97 for Other and 99 for None are
    a house convention that carries meaning into the data tables.
    """
    row_note: str | None = None
    """A directive attached to this row alone, e.g. TERMINATE on an age band.

    Programmer-facing, like routing_notes: shown for reference, never emitted.
    """
    min_value: str | None = None
    max_value: str | None = None
    """A numeric range this row alone accepts, e.g. "min 0 max 200".

    Unlike :attr:`row_note` these *are* emitted: they constrain what the
    respondent may enter, so they belong in the XML rather than in a note to
    the programmer. Both stay ``None`` on a row with no range - a "None of
    these" row in a numeric grid is not a numeric entry at all.
    """

    @classmethod
    def from_runs(cls, runs: list[TextRun], text: str | None = None) -> OptionLine:
        from app.classify.features import detect_trailing_code, strip_trailing_code

        meaningful = [run for run in runs if run.text.strip()]
        source = text if text is not None else "".join(r.text for r in runs).strip()
        return cls(
            raw_text=strip_trailing_code(source),
            code=detect_trailing_code(source),
            bold=bool(meaningful) and all(run.bold for run in meaningful),
            italic=bool(meaningful) and all(run.italic for run in meaningful),
        )

    @classmethod
    def from_text(cls, text: str) -> OptionLine:
        """Build from a plain line, splitting off code and any per-row note.

        A three-column row reads as text | code | note, which is how a
        questionnaire writes "17 or younger  1  TERMINATE".
        """
        cells = [cell.strip() for cell in text.split("|")]
        if len(cells) == 3 and cells[1].isdigit() and cells[2]:
            bounds = numeric_bounds(cells[2])
            if bounds:
                low, high = bounds
                return cls(
                    raw_text=cells[0], code=cells[1], min_value=low, max_value=high
                )
            return cls(raw_text=cells[0], code=cells[1], row_note=cells[2])
        return cls(raw_text=strip_code(text), code=code_of(text))


class ClassificationTrace(BaseModel):
    """The evidence behind one classification, kept for the correction loop."""

    lines: list[str] = Field(default_factory=list)
    ai_payload: dict = Field(default_factory=dict)


class Question(BaseModel):
    """One classified question, ready for review and then generation."""

    label: str
    element: str
    title: list[TextRun] = Field(default_factory=list)
    comment: list[TextRun] = Field(default_factory=list)
    options: list[OptionLine] = Field(default_factory=list)
    rows: list[OptionLine] = Field(default_factory=list)
    cols: list[OptionLine] = Field(default_factory=list)
    subject_type: SubjectType = "none"
    """What the grid's rows describe. Ignored for non-grid elements."""
    comment_resource: str | None = None
    """Resource label to emit as ``${res.X}``.

    ``None`` means the comment is custom text taken from :attr:`comment`, which
    is how a programmer overrides the automatic tag.
    """
    confidence: float = 0.0
    needs_review: bool = True
    ai_notes: str = ""
    dev_notes: str = ""
    """Survey-programmer scratch notes. Never reaches the XML."""
    routing_notes: list[str] = Field(default_factory=list)
    """Programmer-facing lines: skip logic, quotas, terminations, type markers.

    Shown in the review screen for context and never emitted — routing and quota
    logic is the programmer's job, not something the tool guesses at.
    """
    trace: ClassificationTrace | None = None
    """What the model was shown and what it answered, for correction learning."""

    def title_text(self) -> str:
        return "".join(run.text for run in self.title).strip()

    def comment_text(self) -> str:
        return "".join(run.text for run in self.comment).strip()

    @property
    def is_grid(self) -> bool:
        return self.element in GRID_ELEMENTS

    @property
    def is_question(self) -> bool:
        """False when this block is programmer content, not a question."""
        return self.element not in NON_QUESTION_ELEMENTS

    @property
    def generates_xml(self) -> bool:
        return self.element not in NO_XML_ELEMENTS


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


def code_of(text: str) -> str | None:
    from app.classify.features import detect_trailing_code

    return detect_trailing_code(text)


def strip_code(text: str) -> str:
    from app.classify.features import strip_trailing_code

    return strip_trailing_code(text)


def numeric_bounds(text: str) -> tuple[str, str] | None:
    from app.classify.features import detect_numeric_bounds

    return detect_numeric_bounds(text)
