"""Row and column labelling rules, ported from `decipher-subl.py`.

The r91/r99 conventions are house style: "Other (please specify)" is always
r91 with an open text box, "None of the above" is always r99 and never
randomises. Keeping them here rather than inline in the templates means the
XML shapes stay declarative.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.models.survey import OptionLine

#: "Other, please specify" — both words must be present.
_OTHER = re.compile(r"\bother\b", re.IGNORECASE)
_SPECIFY = re.compile(r"\bspecify\b", re.IGNORECASE)

#: "None of the above" / "None of these".
_NONE_OF = re.compile(r"none\s+of\s+(?:the\s+above|these)", re.IGNORECASE)

OTHER_LABEL_SUFFIX = 91
NONE_LABEL_SUFFIX = 99

#: Attributes added to an "other, please specify" row.
OPEN_ATTRS = {"open": "1", "openSize": "25", "randomize": "0"}


def is_other_specify(text: str) -> bool:
    return bool(_OTHER.search(text) and _SPECIFY.search(text))


def is_none_of_the_above(text: str) -> bool:
    return bool(_NONE_OF.search(text))


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


def label_rows(options: list[OptionLine], *, element: str) -> list[LabelledLine]:
    """Assign row labels, preferring codes the source document supplied.

    Precedence is explicit code, then the r91/r99 convention, then sequential.
    The sequential counter only advances for rows that fall through to it, so
    ``[A, Other specify, B]`` produces ``r1, r91, r2``.
    """
    is_checkbox = element in {"checkbox", "checkbox_grid"}
    labelled: list[LabelledLine] = []
    counter = 0

    # Suffixes the source already claimed. Sequential numbering has to step
    # over them: a coded "Male | 1" beside an uncoded "Other" was handing both
    # the label r1, which is two rows with the same identity.
    claimed = {code for code in (_explicit_code(option) for option in options) if code}

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

        # A range the source stated for this row alone. Unlike a row_note this
        # constrains the answer, so it is emitted rather than shown for
        # reference; a row without one is left unconstrained.
        if option.min_value is not None and option.max_value is not None:
            attrs["min"] = option.min_value
            attrs["max"] = option.max_value

        # Numbering precedence: a code the source gave, then the r91/r99
        # convention, then sequential. A source that codes Other as 97 means it;
        # renumbering it to 3 would break the data tables downstream.
        explicit = _explicit_code(option)
        if explicit is not None:
            suffix = explicit
        elif is_other_specify(text):
            suffix = OTHER_LABEL_SUFFIX
        elif is_none_of_the_above(text):
            suffix = NONE_LABEL_SUFFIX
        else:
            counter += 1
            while counter in claimed:
                counter += 1
            suffix = counter

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
