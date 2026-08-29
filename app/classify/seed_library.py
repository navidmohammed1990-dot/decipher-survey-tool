"""Curated examples the classifier always carries.

Distinct from :mod:`app.classify.corrections`, which learns within one document
and is cleared between them. These are patterns worth recognising on first
sight, in every session, without waiting for someone to correct them again.

Kept deliberately small. Every entry is prepended to every classification call,
so it costs prompt tokens on each one — and prompt size is paid for in seconds
on a CPU-bound runtime. Add an entry only when the pattern is one the model
reliably gets wrong without it.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field


class SeedExample(BaseModel):
    """One worked example: what the lines look like, and the right answer."""

    name: str
    lines: list[str] = Field(default_factory=list)
    correct: dict = Field(default_factory=dict)
    why: str = ""

    def as_prompt_example(self) -> str:
        return (
            f"Example - {self.name}:\n"
            f"Lines: {json.dumps(self.lines)}\n"
            f"Correct answer: {json.dumps(self.correct, sort_keys=True)}\n"
            f"Why: {self.why}"
        )


#: A derived-variable definition. Encountered as a real paste, where the tool
#: had no "this is not a question" outcome and forced it into a radio: one row
#: survived, and that row was the variable's own name.
DERIVED_VARIABLE = SeedExample(
    name="derived variable definition, not a question",
    lines=[
        "[Please create the following variable for datafile and auto code based on S2 Age]:",
        "S2_AGE BANDS",
        "Under 18 years | 1",
        "70 years or more | 8",
    ],
    correct={"element": "not_a_question"},
    why=(
        "Speaks to the programmer, refers to an already-asked question (S2), names "
        "a variable rather than asking anything. The bands are a code frame."
    ),
)

SEED_EXAMPLES: tuple[SeedExample, ...] = (DERIVED_VARIABLE,)


def prompt_prefix() -> str:
    """Seed examples to prepend to the system prompt, or an empty string."""
    if not SEED_EXAMPLES:
        return ""
    body = "\n\n".join(example.as_prompt_example() for example in SEED_EXAMPLES)
    return f"Worked examples of patterns that are easy to misread:\n\n{body}\n\n"
