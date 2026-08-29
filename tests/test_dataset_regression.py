"""Phase 12 — the reference dataset, run as a regression suite.

Every entry in ``reference/question_examples.yaml`` is checked on what can be
verified without a live model: that the parser recovers each expected option
text and code, and that the generator turns those into the labels and
attributes the template calls for.

Which *element* the model chooses is judgment and needs a real runtime, so it
is not asserted here. Entries that cannot pass are listed in
:data:`EXPECTED_FAILURES` with a reason, so the rest guard against regression
now rather than waiting for full coverage.
"""

from __future__ import annotations

import pytest

from app.dataset import Example, check, check_generation, check_parsing, load_examples

#: id -> why it cannot pass yet. An entry here that starts passing is itself a
#: failure: the note is stale and should be removed.
EXPECTED_FAILURES: dict[str, str] = {
    "checkbox_grid_basic": (
        "Dataset expects ${res.MR}, but the Phase 5 addendum maps checkbox_grid "
        "to the ${res.MRStatement} variant. One of the two needs correcting - "
        "this is a specification conflict, not a code defect."
    ),
}

EXAMPLES = load_examples()
IDS = [example.id for example in EXAMPLES]


def test_the_dataset_loads():
    assert len(EXAMPLES) >= 19
    assert len({e.id for e in EXAMPLES}) == len(EXAMPLES), "ids must be unique"


def test_every_entry_has_an_input_and_an_expectation():
    for example in EXAMPLES:
        assert example.input.strip(), f"{example.id} has no input"
        assert example.expected, f"{example.id} has no expectation"


@pytest.mark.parametrize("example", EXAMPLES, ids=IDS)
def test_dataset_entry(example: Example):
    """Each entry, checked on everything that does not need a model."""
    problems = check(example)
    reason = EXPECTED_FAILURES.get(example.id)

    if reason:
        if not problems:
            pytest.fail(
                f"{example.id} now passes — remove it from EXPECTED_FAILURES.\n"
                f"Recorded reason was: {reason}"
            )
        pytest.xfail(f"{example.id}: {reason}")

    assert not problems, f"{example.id}:\n" + "\n".join(f"  - {p}" for p in problems)


def test_expected_failures_are_all_real_entries():
    """A stale id in the list would silently excuse nothing."""
    assert set(EXPECTED_FAILURES) <= set(IDS)


def test_most_of_the_dataset_passes():
    """A coverage figure, so a slide backwards is visible at a glance."""
    failing = {e.id for e in EXAMPLES if check(e)}
    unexpected = failing - set(EXPECTED_FAILURES)

    assert not unexpected, f"newly failing: {sorted(unexpected)}"
    assert len(EXAMPLES) - len(failing) >= 18, "coverage dropped"


# -- the patterns Phase 13 added, pinned individually ----------------------


def by_id(example_id: str) -> Example:
    return next(e for e in EXAMPLES if e.id == example_id)


def test_two_table_grid_recovers_scale_and_statements():
    """Columns arrive on one line and their codes on the next."""
    assert check_parsing(by_id("two_table_grid")) == []


def test_per_row_routing_column_keeps_its_directive():
    from app.dataset import parsed_lines
    from app.models.survey import OptionLine

    lines = parsed_lines(by_id("per_row_routing_column"))
    notes = {
        OptionLine.from_text(line.text).raw_text: OptionLine.from_text(line.text).row_note
        for line in lines
    }
    assert notes.get("17 or younger") == "TERMINATE"
    assert notes.get("70 or older") == "TERMINATE"
    assert notes.get("18-24") is None


def test_strikethrough_entry_is_all_struck():
    from app.dataset import parsed_lines

    lines = parsed_lines(by_id("strikethrough_excluded"))
    assert lines and all(line.features.is_struck for line in lines)


def test_missing_code_falls_back_without_colliding():
    assert check_generation(by_id("missing_code_on_one_option")) == []


def test_non_sequential_codes_are_not_renumbered():
    assert check_parsing(by_id("non_sequential_codes_preserved")) == []


def test_wrapped_option_is_recovered_whole():
    assert check_parsing(by_id("multiline_wrapped_option")) == []
