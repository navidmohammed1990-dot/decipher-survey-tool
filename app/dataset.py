"""The reference dataset: real input patterns and what they should produce.

Loaded from ``reference/question_examples.yaml`` and used two ways — as a
regression suite, and as a source of few-shot examples.

What the harness can check without a model, it checks: whether the parser
recovers every expected option text and code, and whether the generator turns
those into the labels and attributes the template calls for. Which *element*
the model picks is judgment, so that part is only checkable against a live
runtime and is reported separately rather than faked.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from app.classify.paste import split_questions
from app.generate.xml_generator import generate_question
from app.models.document import TextRun
from app.models.survey import NO_XML_ELEMENTS, OptionLine, Question

DATASET_PATH = Path(__file__).resolve().parent.parent / "reference" / "question_examples.yaml"


class Example(BaseModel):
    """One dataset entry."""

    id: str
    category: str = ""
    description: str = ""
    input: str = ""
    expected: dict = Field(default_factory=dict)

    @property
    def element(self) -> str | None:
        return self.expected.get("element")

    def expected_lines(self, key: str) -> list[dict]:
        """The expected rows/options/cols, normalised to dicts."""
        return [entry for entry in (self.expected.get(key) or []) if isinstance(entry, dict)]

    @property
    def answer_entries(self) -> list[dict]:
        """Whichever key this entry uses for its answer list."""
        return self.expected_lines("rows") or self.expected_lines("options")


def load_examples(path: str | Path | None = None) -> list[Example]:
    """Read the dataset, newest schema wins if a key is missing."""
    source = Path(path or DATASET_PATH)
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or []
    return [Example.model_validate(entry) for entry in raw if isinstance(entry, dict)]


def parsed_lines(example: Example) -> list:
    """The source lines the classifier would be shown for this entry."""
    blocks, _ = split_questions(example.input)
    return [line for block in blocks for line in block.lines]


def _text_and_code(line) -> tuple[str, str | None]:
    option = OptionLine.from_text(line.text)
    return option.raw_text, option.code


def check_parsing(example: Example) -> list[str]:
    """Every expected answer text must survive parsing, with its code."""
    failures: list[str] = []
    found = dict(_text_and_code(line) for line in parsed_lines(example))

    for entry in example.answer_entries + example.expected_lines("cols"):
        text = entry.get("text")
        if not text:
            continue
        if text not in found:
            failures.append(f"option text not recovered: {text!r}")
            continue

        wanted = entry.get("code") if "code" in entry else entry.get("value")
        if wanted is not None and found[text] != str(wanted):
            failures.append(
                f"{text!r} code was {found[text]!r}, expected {str(wanted)!r}"
            )
    return failures


def build_question(example: Example) -> Question:
    """A Question carrying the entry's expected element and answer list.

    The element comes from the expectation rather than from a classifier, so
    this exercises the deterministic half of the pipeline on its own.
    """
    entries = example.answer_entries
    options = [
        OptionLine(
            raw_text=entry["text"],
            code=str(entry["code"]) if entry.get("code") is not None else
                 (str(entry["value"]) if entry.get("value") is not None else None),
        )
        for entry in entries if entry.get("text")
    ]
    cols = [
        OptionLine(
            raw_text=entry["text"],
            code=str(entry["code"]) if entry.get("code") is not None else None,
        )
        for entry in example.expected_lines("cols") if entry.get("text")
    ]

    element = example.element or "radio"
    is_grid = element in ("radio_grid", "checkbox_grid")
    title = example.expected.get("title") or example.id

    return Question(
        label="Q1",
        element=element,
        title=[TextRun(text=str(title))],
        options=[] if is_grid else options,
        rows=options if is_grid else [],
        cols=cols,
        needs_review=False,
    )


def check_generation(example: Example) -> list[str]:
    """The generated XML must carry the labels and attributes expected."""
    if example.element in NO_XML_ELEMENTS or example.element is None:
        return []

    failures: list[str] = []
    xml = generate_question(build_question(example))

    tag = example.expected.get("comment_tag")
    if tag and f"<comment>{tag}</comment>" not in xml:
        failures.append(f"comment tag {tag!r} missing")

    if example.expected.get("atleast") and 'atleast="1"' not in xml:
        failures.append('atleast="1" missing')

    for entry in example.expected_lines("rows"):
        label = entry.get("label")
        if label and f'label="{label}"' not in xml:
            failures.append(f"row label {label!r} missing")
        if entry.get("open") and 'open="1"' not in xml:
            failures.append(f"open attribute missing on {label}")
        if entry.get("exclusive") and 'exclusive="1"' not in xml:
            failures.append(f"exclusive attribute missing on {label}")

    return failures


def check(example: Example) -> list[str]:
    """Everything checkable without a live model."""
    return check_parsing(example) + check_generation(example)
