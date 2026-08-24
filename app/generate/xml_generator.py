"""Phase 3 — the deterministic Decipher XML generator.

Zero AI involvement. Given the same :class:`Question`, this module always
produces byte-for-byte identical output: no model call, no randomness, no
timestamps. That is what makes the generated base safe to trust.

The attribute sets below are ported verbatim from the team's canonical
template. They are data, not suggestions — do not "improve" them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.generate.labels import LabelledLine, label_cols, label_rows
from app.generate.text import clean, option_markup, runs_to_markup
from app.models.survey import SUPPORTED_ELEMENTS, Question

INDENT = "  "
SUSPEND = "<suspend/>"


class UnsupportedElementError(ValueError):
    """Raised for an element outside :data:`SUPPORTED_ELEMENTS`."""


@dataclass(frozen=True)
class ElementSpec:
    """The fixed XML shape for one element type."""

    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    comment_default: str = ""
    validate: str = ""
    """``CheckBlank`` argument, e.g. ``"1"``. Empty means no validate element."""
    has_rows: bool = False
    has_cols: bool = False
    row_values: bool = False
    """True when rows carry ``value="N"`` matching the label suffix."""


#: Verbatim from `Standard_Template_Questions_V1_24Aug26.xml`.
ELEMENT_SPECS: dict[str, ElementSpec] = {
    "radio": ElementSpec(
        tag="radio",
        attrs={
            "atm1d:showInput": "0",
            "atm1d:viewMode": "vertical",
            "randomize": "0",
            "ss:listDisplay": "1",
            "uses": "atm1d.10",
            "values": "order",
        },
        comment_default="${res.SR}",
        validate="1",
        has_rows=True,
        row_values=True,
    ),
    "checkbox": ElementSpec(
        tag="checkbox",
        attrs={
            "atleast": "1",
            "atm1d:showInput": "0",
            "atm1d:viewMode": "vertical",
            "fwidth": "1000",
            "randomize": "0",
            "ss:listDisplay": "1",
            "uses": "atm1d.10",
        },
        comment_default="${res.MR}",
        validate="1",
        has_rows=True,
        # Checkbox rows omit value — matches the canonical template exactly.
        row_values=False,
    ),
    "textarea": ElementSpec(
        tag="textarea",
        attrs={"height": "10", "optional": "0", "randomize": "0", "width": "50"},
        comment_default="${res.Open}",
        validate="2",
    ),
    "text": ElementSpec(
        tag="text",
        attrs={"optional": "0", "randomize": "0", "size": "25"},
        comment_default="${res.Open}",
        validate="2",
    ),
    "number": ElementSpec(
        tag="number",
        attrs={"size": "3", "optional": "0"},
        comment_default="Please enter a whole number",
    ),
    "select": ElementSpec(
        tag="select",
        attrs={"optional": "0"},
        has_rows=True,
    ),
    "radio_grid": ElementSpec(
        tag="radio",
        attrs={"randomize": "0"},
        comment_default="${res.SR}",
        has_rows=True,
        has_cols=True,
    ),
    "checkbox_grid": ElementSpec(
        tag="checkbox",
        attrs={"atleast": "1", "randomize": "0"},
        comment_default="${res.MR}",
        has_rows=True,
        has_cols=True,
    ),
    "html": ElementSpec(tag="html", attrs={"where": "survey"}),
}

assert set(ELEMENT_SPECS) == set(SUPPORTED_ELEMENTS), "spec table must cover every element"


def _attr_value(value: str) -> str:
    return value.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")


def _open_tag(tag: str, label: str, attrs: dict[str, str]) -> str:
    """``label`` always leads; the rest follow in canonical template order."""
    rendered = "".join(f' {name}="{_attr_value(value)}"' for name, value in attrs.items())
    return f'<{tag} label="{_attr_value(label)}"{rendered}>'


def _line_element(tag: str, line: LabelledLine, *, with_value: bool) -> str:
    attrs = {"label": line.label}
    if with_value:
        attrs["value"] = str(line.suffix)
    attrs.update(line.attrs)

    rendered = "".join(f' {name}="{_attr_value(value)}"' for name, value in attrs.items())
    body = option_markup(line.option.raw_text, line.option.bold, line.option.italic)
    return f"{INDENT}<{tag}{rendered}>{body}</{tag}>"


def _rows_for(question: Question, spec: ElementSpec) -> list[LabelledLine]:
    """Grids read from ``rows``; flat elements read from ``options``."""
    source = question.rows if spec.has_cols else question.options
    return label_rows(source, element=question.element)


def generate_question(question: Question) -> str:
    """Render one question as a Decipher XML fragment.

    Namespace prefixes (``atm1d:``, ``ss:``) are used but not declared — they
    belong on the survey root, not on every element.
    """
    if question.element not in ELEMENT_SPECS:
        raise UnsupportedElementError(
            f"'{question.element}' is not a supported element. "
            f"Expected one of: {', '.join(SUPPORTED_ELEMENTS)}."
        )

    spec = ELEMENT_SPECS[question.element]
    title = runs_to_markup(question.title)

    if question.element == "html":
        return f"{_open_tag(spec.tag, question.label, spec.attrs)}{title}</{spec.tag}>"

    lines = [_open_tag(spec.tag, question.label, spec.attrs)]
    lines.append(f"{INDENT}<title>{title}</title>")

    comment = runs_to_markup(question.comment) or clean(spec.comment_default)
    if comment:
        lines.append(f"{INDENT}<comment>{comment}</comment>")

    if spec.has_rows:
        for row in _rows_for(question, spec):
            lines.append(_line_element("row", row, with_value=spec.row_values))

    if spec.has_cols:
        for col in label_cols(question.cols):
            lines.append(_line_element("col", col, with_value=False))

    if spec.validate:
        lines.append(
            f"{INDENT}<validate>CheckBlank({spec.validate},{question.label})</validate>"
        )

    lines.append(f"</{spec.tag}>")
    return "\n".join(lines)


def generate_fragment(question: Question) -> str:
    """One question plus the blank line and ``<suspend/>`` that follow it."""
    return f"{generate_question(question)}\n\n{SUSPEND}"


def generate_questions(questions: list[Question]) -> str:
    """Assemble the base XML for a whole questionnaire."""
    return "\n\n".join(generate_fragment(question) for question in questions)
