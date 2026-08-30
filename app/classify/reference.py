"""The reference dataset, as precedent for live classification.

`reference/question_examples.yaml` began as a regression suite: 27 real input
patterns paired with the classification each should get. That made it a test of
the deterministic half of the pipeline, and nothing more - the model never saw
any of it.

It is worth more than that. Each entry is a real question somebody already
decided the right answer for, which is exactly what a few-shot example is. So
the entries relevant to the question in front of the model are now sent with
the call, alongside the programmer's own corrections.

Precedent, never override. These reach the model the same way Phase 7's
formatting hints do - as something to weigh against its own reading of the
actual input, not as a rule that settles it.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

#: How many dataset examples may accompany one call. The dataset is meant to
#: grow; the prompt is not.
MAX_REFERENCE_EXAMPLES = 2

_cache: list[tuple[list[str], str]] | None = None


def _build() -> list[tuple[list[str], str]]:
    """Every scored entry as (source lines, rendered prompt example)."""
    from app.classify.paste import split_questions
    from app.dataset import load_examples

    built: list[tuple[list[str], str]] = []
    for example in load_examples():
        if not example.input.strip() or not example.element:
            continue
        try:
            blocks, _ = split_questions(example.input)
        except Exception:  # a malformed entry must not break classification
            logger.warning("Reference entry %s could not be parsed", example.id)
            continue

        lines = [line.text for block in blocks for line in block.lines]
        if not lines:
            continue

        correct: dict = {"element": example.element}
        subject = example.expected.get("subject_type")
        if subject:
            correct["subject_type"] = subject

        built.append((
            lines,
            f"Example - {example.id}:\n"
            f"Lines: {json.dumps(lines)}\n"
            f"Correct answer: {json.dumps(correct, sort_keys=True)}\n"
            f"Why: {example.description or example.category}",
        ))
    return built


def examples() -> list[tuple[list[str], str]]:
    """The dataset, parsed once and kept."""
    global _cache
    if _cache is None:
        _cache = _build()
    return _cache


def reset_cache() -> None:
    """Forget the parsed dataset, so a test can point at a different file."""
    global _cache
    _cache = None


def prompt_prefix(lines: list[str], limit: int = MAX_REFERENCE_EXAMPLES) -> str:
    """The most relevant known examples, rendered for the system prompt.

    Empty when nothing in the dataset resembles this question - an unrelated
    example is worse than none, since it spends prompt tokens describing a
    question the model was not asked.
    """
    from app.classify.relevance import most_relevant

    chosen = most_relevant(
        lines, [(text, rendered) for text, rendered in examples()], limit=limit
    )
    if not chosen:
        return ""
    body = "\n\n".join(str(entry) for entry in chosen)
    return (
        "Questions like this one, and the answer each was given "
        f"(precedent, not a rule - judge the lines you were sent):\n\n{body}\n\n"
    )
