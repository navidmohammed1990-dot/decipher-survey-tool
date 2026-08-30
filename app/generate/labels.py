"""Row and column labelling rules, ported from `decipher-subl.py`.

The r91/r99 conventions are house style: "Other (please specify)" is always
r91 with an open text box, "None of the above" is always r99 and never
randomises. Keeping them here rather than inline in the templates means the
XML shapes stay declarative.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from app.models.survey import OptionLine

#: "Other, please specify" — both words must be present.
_OTHER = re.compile(r"\bother\b", re.IGNORECASE)
_SPECIFY = re.compile(r"\bspecify\b", re.IGNORECASE)

#: "None of the above" / "None of these".
_NONE_OF = re.compile(r"none\s+of\s+(?:the\s+above|these)", re.IGNORECASE)

#: "Don't know" / "Do not know" / "Not sure".
#:
#: Anchored at the start, unlike the two above. A row that offers this as an
#: answer says so first; "I don't know how often I shop there" is a real
#: answer that happens to contain the words, and must not be read as one.
_DONT_KNOW = re.compile(
    r"^(?:don'?t\s+know|do\s+not\s+know|not\s+sure|unsure)\b", re.IGNORECASE
)

#: "Prefer not to say", and the same thought written as a refusal.
_PREFER_NOT = re.compile(
    r"^(?:prefer\s+not\s+to\s+(?:say|answer)|rather\s+not\s+say)\b", re.IGNORECASE
)

#: "Not applicable" / "N/A", and a "None…" that is not "none of the above".
#: These have no default code of their own - they only mark a row as an
#: opt-out for :func:`is_opt_out`.
_OTHER_OPT_OUT = re.compile(
    r"^(?:none\b|not\s+applicable\b|n/?a\b)", re.IGNORECASE
)

OTHER_LABEL_SUFFIX = 91
DONT_KNOW_LABEL_SUFFIX = 97
PREFER_NOT_LABEL_SUFFIX = 98
NONE_LABEL_SUFFIX = 99

#: Attributes added to an "other, please specify" row.
OPEN_ATTRS = {"open": "1", "openSize": "25", "randomize": "0"}


def is_other_specify(text: str) -> bool:
    """"Other, please specify" - both words, and little else in the row.

    The second half is Phase 20b: "Please specify any other brands you have
    used" has both words and is a question, not an other-specify row.
    """
    stripped = text.strip()
    return bool(
        _OTHER.search(stripped)
        and _SPECIFY.search(stripped)
        and _is_the_whole_row(stripped)
    )


def is_none_of_the_above(text: str) -> bool:
    """"None of the above" as the row's whole content, not as its opening.

    "None of the above brands appeal to me" is an answer about brands; reading
    it as the None row would mark a real answer exclusive.
    """
    stripped = text.strip()
    return bool(_NONE_OF.search(stripped)) and _is_the_whole_row(stripped)


#: Every way of writing one of the four special categories, for the "is that
#: all this row says?" test below. Wider than any single category on purpose:
#: a row reading "Don't know / Can't say" is still only an opt-out.
_CATEGORY_PHRASE = re.compile(
    r"don'?t\s+know|do\s+not\s+know|not\s+sure|unsure"
    r"|prefer\s+not\s+to\s+(?:say|answer)|rather\s+not\s+say"
    r"|can'?t\s+say|cannot\s+say|no\s+opinion"
    r"|none\s+of\s+(?:the\s+above|these)|not\s+applicable|n/a"
    r"|other|please|specify|below",
    re.IGNORECASE,
)

#: Words that carry no subject of their own: joiners, and the generic tails a
#: row adds to finish the sentence. "None of these apply" and "None of the
#: above apply to me" are the None row; "None of the above brands appeal to me"
#: is about brands, and is not.
#:
#: A closed vocabulary, like the phrases above, and the same caveat applies -
#: it covers the house styles it was written against. Being wrong here costs a
#: special row its house code and a plainly numbered row appears instead, which
#: is visible in review; being wrong the other way silently marks a real answer
#: exclusive.
_FILLER = re.compile(
    r"\b(?:or|and|apply|applies|applicable|relevant|to\s+me|for\s+me"
    r"|of\s+(?:the\s+above|these)|above|these|this|that|it|them)\b",
    re.IGNORECASE,
)


def _is_the_whole_row(text: str) -> bool:
    """Whether category phrasing is all this row says.

    Anchoring at the start is not enough on its own: "Not sure why the parcel
    was late, but it arrived" opens with the words and is an ordinary answer,
    as is "None of the above brands appeal to me". A row that *is* one of these
    categories has nothing else in it, give or take punctuation and filler.
    """
    return not re.search(r"[A-Za-z0-9]", _FILLER.sub(" ", _CATEGORY_PHRASE.sub(" ", text)))


def is_dont_know(text: str) -> bool:
    stripped = text.strip()
    return bool(_DONT_KNOW.match(stripped)) and _is_the_whole_row(stripped)


def is_prefer_not_to_say(text: str) -> bool:
    stripped = text.strip()
    return bool(_PREFER_NOT.match(stripped)) and _is_the_whole_row(stripped)


#: The house code each special category gets *when the source gives none*.
#: Order matters only for a row that reads as two categories at once; the
#: first match wins, and these are written most-specific first.
#:
#: This is the whole of the convention, in one place. A source code always
#: outranks it - see :func:`_explicit_code` and the precedence in
#: :func:`label_rows`.
DEFAULT_CODES: tuple[tuple[int, Callable[[str], bool]], ...] = (
    (OTHER_LABEL_SUFFIX, is_other_specify),
    (NONE_LABEL_SUFFIX, is_none_of_the_above),
    (DONT_KNOW_LABEL_SUFFIX, is_dont_know),
    (PREFER_NOT_LABEL_SUFFIX, is_prefer_not_to_say),
)


def default_code(text: str) -> int | None:
    """The house code this option gets if the source gave it none."""
    for suffix, matches in DEFAULT_CODES:
        if matches(text):
            return suffix
    return None


def is_opt_out(text: str) -> bool:
    """Whether this row offers a way out of answering rather than an answer.

    Broader than the categories that carry a default code: "Not applicable"
    and a bare "None, I have not taken any..." are opt-outs too, they just
    have no house code of their own.

    A vocabulary, and therefore the kind of thing that only ever covers the
    house styles it was written against. In a numeric grid it is the second
    opinion, not the test: a row that states no range where its siblings do is
    already an opt-out on structure alone.
    """
    if is_dont_know(text) or is_prefer_not_to_say(text) or is_none_of_the_above(text):
        return True
    return bool(_OTHER_OPT_OUT.match(text.strip()))


@dataclass
class LabelledLine:
    """One row or column with its resolved label and extra attributes."""

    option: OptionLine
    label: str
    suffix: int
    attrs: dict[str, str]


def _explicit_code(option: OptionLine) -> int | None:
    """A source-provided code, if the option carries a usable one."""
    if not option.code:
        return None
    try:
        return int(option.code)
    except (TypeError, ValueError):
        return None


#: Marks a slider point that opts out of the scale rather than sitting on it.
SLIDER_OPT_OUT_ATTRS = {"sliderpoints:OO": "1"}


def label_choices(options: list[OptionLine]) -> list[LabelledLine]:
    """Label a slider's scale points ``ch1``, ``ch2`` … in source order.

    The same numbering precedence as rows - a source code first, then the
    house default, then sequential - with one addition: an opt-out point that
    the house table has no code for takes 99, matching the template's
    ``ch99`` for "NA". It is off the scale, so it cannot take the next number
    on it.
    """
    labelled: list[LabelledLine] = []
    counter = 0
    claimed = {code for code in (_explicit_code(option) for option in options) if code}
    taken = set(claimed)

    for option in options:
        text = option.raw_text
        opt_out = is_opt_out(text)
        attrs = dict(SLIDER_OPT_OUT_ATTRS) if opt_out else {}

        explicit = _explicit_code(option)
        house = default_code(text)
        if explicit is not None:
            suffix = explicit
        elif house is not None and house not in taken:
            suffix = house
        elif opt_out and NONE_LABEL_SUFFIX not in taken:
            suffix = NONE_LABEL_SUFFIX
        else:
            counter += 1
            while counter in taken:
                counter += 1
            suffix = counter

        taken.add(suffix)
        labelled.append(
            LabelledLine(option=option, label=f"ch{suffix}", suffix=suffix, attrs=attrs)
        )

    return labelled


def label_rows(options: list[OptionLine], *, element: str) -> list[LabelledLine]:
    """Assign row labels, preferring codes the source document supplied.

    Precedence is explicit code, then the r91/r99 convention, then sequential.
    The sequential counter only advances for rows that fall through to it, so
    ``[A, Other specify, B]`` produces ``r1, r91, r2``.
    """
    is_checkbox = element in {"checkbox", "checkbox_grid"}
    labelled: list[LabelledLine] = []
    counter = 0

    # Suffixes the source already claimed. Nothing chosen here may reuse one:
    # a coded "Male | 1" beside an uncoded "Other" was handing both the label
    # r1, which is two rows with the same identity.
    claimed = {code for code in (_explicit_code(option) for option in options) if code}
    taken = set(claimed)

    for option in options:
        text = option.raw_text
        attrs: dict[str, str] = {}

        # Attributes follow what the option *says*, independently of where its
        # number came from: an explicitly coded "Other (97), please specify"
        # still needs its text box.
        if is_other_specify(text):
            attrs.update(OPEN_ATTRS)
        elif is_none_of_the_above(text):
            attrs["randomize"] = "0"
            if is_checkbox:
                # Radio needs no exclusive: only one answer is possible anyway.
                attrs["exclusive"] = "1"
        elif is_dont_know(text) or is_prefer_not_to_say(text):
            # Fixed at the end of the list like None is, confirmed in 20b.
            # No exclusive: only randomize was confirmed, and inventing the
            # rest is how the per-row min/max shape went wrong.
            attrs["randomize"] = "0"

        # Numbering precedence: a code the source gave, then the r91/r99
        # convention, then sequential. A source that codes Other as 97 means it;
        # renumbering it to 3 would break the data tables downstream.
        explicit = _explicit_code(option)
        house = default_code(text)
        if explicit is not None:
            suffix = explicit
        elif house is not None and house not in taken:
            suffix = house
        else:
            # Either an ordinary option, or a special row whose house code the
            # source already spent elsewhere - A9b codes "Don't know" as 98,
            # so an uncoded "Prefer not to say" in the same question cannot
            # also be r98. Two rows with one label is the worse outcome, so it
            # falls through to a free number rather than colliding.
            counter += 1
            while counter in taken:
                counter += 1
            suffix = counter

        taken.add(suffix)

        labelled.append(
            LabelledLine(option=option, label=f"r{suffix}", suffix=suffix, attrs=attrs)
        )

    return labelled


def label_cols(options: list[OptionLine]) -> list[LabelledLine]:
    """Assign ``c1``, ``c2``, … sequentially.

    Columns take the other/specify open-text handling but not the r91/r99
    numbering convention — that is rows only.
    """
    labelled: list[LabelledLine] = []
    counter = 0
    claimed = {code for code in (_explicit_code(option) for option in options) if code}
    for option in options:
        attrs = dict(OPEN_ATTRS) if is_other_specify(option.raw_text) else {}
        explicit = _explicit_code(option)
        if explicit is not None:
            suffix = explicit
        else:
            counter += 1
            while counter in claimed:
                counter += 1
            suffix = counter
        labelled.append(
            LabelledLine(option=option, label=f"c{suffix}", suffix=suffix, attrs=attrs)
        )
    return labelled
