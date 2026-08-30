"""Choosing which known examples are worth sending with a classification call.

Two knowledge sources feed the prompt - the curated reference dataset and the
programmer's own corrections - and both grow. The dataset grows as real cases
are added; the correction library grows every time someone corrects a question.
Sending all of either would make the prompt bigger every month, and on a
CPU-bound runtime prompt size is paid for in seconds on every call.

So examples are chosen, not accumulated: the few most like the question in
front of the model, capped, so the prompt stays the same size whether the
sources hold ten examples or ten thousand.

The signal is deliberately plain - shared wording and a similar shape. It only
has to keep an unrelated grid example out of an open-text classification; it is
not, and does not need to be, semantic search.
"""

from __future__ import annotations

import re

#: Words too common to say anything about what a question is about.
_STOPWORDS = frozenset(
    """a an and are as at be by do does for from has have how i if in is it its
    of on or please that the this to was were what when which who why will with
    you your select all apply following each""".split()
)

_WORD = re.compile(r"[a-z0-9']+")

#: How much of the score comes from wording rather than shape. Wording is the
#: stronger signal - two questions about brands read alike - but shape catches
#: the case wording misses: a grid looks like a grid whatever it asks about.
_WORDING_WEIGHT = 0.7


def _words(lines: list[str]) -> set[str]:
    text = " ".join(lines).lower()
    return {word for word in _WORD.findall(text) if word not in _STOPWORDS and len(word) > 2}


def _shape(lines: list[str]) -> tuple[int, float]:
    """How many lines, and what share of them carry a code."""
    from app.classify.features import detect_trailing_code

    if not lines:
        return 0, 0.0
    coded = sum(1 for line in lines if detect_trailing_code(line))
    return len(lines), coded / len(lines)


def score(current: list[str], candidate: list[str]) -> float:
    """How much ``candidate`` is worth showing when classifying ``current``.

    Between 0 and 1. Wording overlap carries most of it; the rest rewards a
    similar shape, so a five-option coded list is matched by another one even
    when the two ask about completely different subjects.
    """
    if not current or not candidate:
        return 0.0

    here, there = _words(current), _words(candidate)
    shared = len(here & there) / len(here | there) if here | there else 0.0

    lines_here, coded_here = _shape(current)
    lines_there, coded_there = _shape(candidate)
    longest = max(lines_here, lines_there, 1)
    length = 1.0 - abs(lines_here - lines_there) / longest
    coding = 1.0 - abs(coded_here - coded_there)

    return _WORDING_WEIGHT * shared + (1 - _WORDING_WEIGHT) * (length + coding) / 2


def most_relevant(
    current: list[str],
    candidates: list[tuple[list[str], object]],
    *,
    limit: int,
    floor: float = 0.2,
) -> list[object]:
    """The ``limit`` candidates most like ``current``, best first.

    ``floor`` keeps a weak match out entirely: an example with nothing in
    common is worse than no example, because it spends prompt tokens telling
    the model about a question it was not asked.
    """
    ranked = sorted(
        ((score(current, lines), payload) for lines, payload in candidates),
        key=lambda pair: pair[0],
        reverse=True,
    )
    return [payload for value, payload in ranked[:limit] if value >= floor]
