"""Phase 14 — plain-English corrections, alongside the existing fields."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.classify import commands
from app.classify.ollama import OllamaClient, OllamaError
from app.main import app
from app.models.document import TextRun
from app.models.survey import OptionLine, Question


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _clean():
    from app.api.routes_quick import quick_corrections

    quick_corrections.clear()
    yield
    quick_corrections.clear()


def a_question(**kwargs):
    defaults = dict(
        label="Q1",
        element="checkbox",
        title=[TextRun(text="Which of these apply?")],
        options=[OptionLine.from_text(t) for t in ("Yes | 1", "No | 2", "Maybe | 3")],
        confidence=0.9,
        needs_review=False,
    )
    defaults.update(kwargs)
    return Question(**defaults)


class Stub(OllamaClient):
    """A model that returns one canned interpretation."""

    def __init__(self, payload=None, error=None):
        super().__init__()
        self.payload = payload
        self.error = error
        self.prompts: list[str] = []

    def generate_json(self, system, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.payload


@pytest.fixture
def stub_command(monkeypatch):
    def install(payload=None, error=None):
        from app.api import routes_classify

        instance = Stub(payload, error)
        monkeypatch.setattr(routes_classify, "build_client", lambda: instance)
        return instance

    return install


# -- the interpreter -------------------------------------------------------


def test_the_example_instruction_sets_the_element_and_keeps_the_options():
    """Checklist: the worked example from the brief."""
    result = commands.interpret(
        a_question(),
        "Q1 needs to be converted as a radio question with all the options shown below "
        "as radio options.",
        Stub({"understood": True, "changes": {"element": "radio"},
              "reason": "element changed to radio; options left as they are"}),
    )

    assert result.understood is True
    assert result.question.element == "radio"
    assert [o.raw_text for o in result.question.options] == ["Yes", "No", "Maybe"]
    assert [o.code for o in result.question.options] == ["1", "2", "3"], "codes survive"
    assert [c["field"] for c in result.changes] == ["element"]


def test_the_diff_shows_before_and_after():
    result = commands.interpret(
        a_question(), "make it a radio",
        Stub({"understood": True, "changes": {"element": "radio"}, "reason": "ok"}),
    )
    (change,) = result.changes

    assert change == {"field": "element", "before": "checkbox", "after": "radio"}


def test_nothing_is_saved_by_interpreting():
    """The proposal is a copy; the original is untouched until confirmed."""
    question = a_question()
    result = commands.interpret(
        question, "make it a radio",
        Stub({"understood": True, "changes": {"element": "radio"}, "reason": "ok"}),
    )

    assert question.element == "checkbox", "the original is unchanged"
    assert result.question is not question


def test_options_can_be_replaced_explicitly():
    result = commands.interpret(
        a_question(), "the options should be Red, Green and Blue",
        Stub({"understood": True, "changes": {"options": ["Red", "Green", "Blue"]},
              "reason": "replaced options"}),
    )
    assert [o.raw_text for o in result.question.options] == ["Red", "Green", "Blue"]


def test_a_replaced_option_keeps_any_code_it_carries():
    result = commands.interpret(
        a_question(), "options are Male 1 and Female 2",
        Stub({"understood": True, "changes": {"options": ["Male | 1", "Female | 2"]},
              "reason": "set options"}),
    )
    assert [(o.raw_text, o.code) for o in result.question.options] == [
        ("Male", "1"), ("Female", "2"),
    ]


def test_a_title_change_is_applied():
    result = commands.interpret(
        a_question(), "retitle it",
        Stub({"understood": True, "changes": {"title": "A better question?"},
              "reason": "retitled"}),
    )
    assert result.question.title_text() == "A better question?"


def test_an_explicit_comment_switches_off_the_resource_tag():
    result = commands.interpret(
        a_question(comment_resource="MR"), "set the comment to say answer honestly",
        Stub({"understood": True, "changes": {"comment": "Answer honestly."},
              "reason": "custom comment"}),
    )
    assert result.question.comment_resource is None
    assert result.question.comment_text() == "Answer honestly."


# -- refusing to guess -----------------------------------------------------


def test_an_ambiguous_instruction_is_refused():
    """Checklist: a clear "couldn't interpret this", not a silent bad edit."""
    result = commands.interpret(
        a_question(), "do the thing with the stuff",
        Stub({"understood": False, "changes": {}, "reason": "The instruction is not specific."}),
    )

    assert result.understood is False
    assert result.question is None
    assert result.changes == []
    assert "not specific" in result.reason


def test_a_missing_reason_still_points_at_the_fields():
    result = commands.interpret(
        a_question(), "???", Stub({"understood": False, "changes": {}}),
    )
    assert result.reason == commands.UNCLEAR_MESSAGE


def test_an_unknown_element_is_not_applied():
    result = commands.interpret(
        a_question(), "make it a dropdown",
        Stub({"understood": True, "changes": {"element": "dropdown"}, "reason": "dropdown"}),
    )
    assert result.understood is False, "nothing valid was left to apply"


def test_a_no_op_instruction_says_so_rather_than_showing_an_empty_diff():
    result = commands.interpret(
        a_question(), "make it a checkbox",
        Stub({"understood": True, "changes": {"element": "checkbox"}, "reason": "already"}),
    )
    assert result.understood is False
    assert "would not change anything" in result.reason


def test_fields_outside_the_editable_set_are_ignored():
    result = commands.interpret(
        a_question(), "set confidence to 1",
        Stub({"understood": True, "changes": {"confidence": 1.0, "label": "Q99"},
              "reason": "nope"}),
    )
    assert result.understood is False, "neither field is editable by instruction"


def test_an_empty_instruction_is_refused_without_a_model_call():
    stub = Stub({"understood": True, "changes": {"element": "radio"}})
    result = commands.interpret(a_question(), "   ", stub)

    assert result.understood is False
    assert stub.prompts == [], "no round trip is spent on an empty box"


def test_an_unreachable_model_reports_rather_than_crashing():
    result = commands.interpret(
        a_question(), "make it a radio", Stub(error=OllamaError("connection refused"))
    )
    assert result.understood is False
    assert "connection refused" in result.reason


def test_the_prompt_forbids_inventing_content():
    assert "Never invent" in commands.SYSTEM_PROMPT
    assert "Do not guess" in commands.SYSTEM_PROMPT
    assert "understood" in commands.SYSTEM_PROMPT


def test_the_prompt_carries_the_current_state():
    prompt = commands.build_prompt(a_question(), "make it a radio")

    assert "checkbox" in prompt
    assert "Which of these apply?" in prompt
    assert "make it a radio" in prompt


# -- over HTTP -------------------------------------------------------------


def test_the_endpoint_proposes_without_saving(client, stub_command):
    stub_command({"understood": True, "changes": {"element": "radio"}, "reason": "radio"})
    body = client.post("/api/quick-command", json={
        "question": a_question().model_dump(mode="json"),
        "instruction": "make this a radio with the options below",
    }).json()

    assert body["understood"] is True
    assert body["question"]["element"] == "radio"
    assert body["changes"][0]["field"] == "element"


def test_the_endpoint_reports_an_unclear_instruction(client, stub_command):
    stub_command({"understood": False, "changes": {}, "reason": "Too vague."})
    body = client.post("/api/quick-command", json={
        "question": a_question().model_dump(mode="json"),
        "instruction": "asdf",
    }).json()

    assert body["understood"] is False
    assert body["question"] is None
    assert body["reason"] == "Too vague."


def test_an_empty_instruction_is_a_clear_error(client, stub_command):
    stub_command({"understood": True, "changes": {}})
    response = client.post("/api/quick-command", json={
        "question": a_question().model_dump(mode="json"), "instruction": "  ",
    })
    assert response.status_code == 400


def test_confirming_records_a_correction(client, stub_command):
    """Checklist: a confirmed command lands in the library like a manual one."""
    before = a_question()
    after = a_question(element="radio")

    body = client.post("/api/quick-command/confirm", json={
        "before": before.model_dump(mode="json"),
        "after": after.model_dump(mode="json"),
        "instruction": "make this a radio",
    }).json()

    assert '<radio label="Q1"' in body["xml"]

    library = client.get("/api/quick-corrections").json()
    assert library["count"] == 1
    entry = library["corrections"][0]
    assert entry["ai_said"]["element"] == "checkbox"
    assert entry["sp_corrected_to"]["element"] == "radio"
    assert entry["sp_corrected_to"]["instruction"] == "make this a radio"


def test_a_confirmed_correction_reaches_later_classifications(client, stub_command, monkeypatch):
    """The recorded correction becomes context for the next paste."""
    client.post("/api/quick-command/confirm", json={
        "before": a_question().model_dump(mode="json"),
        "after": a_question(element="radio").model_dump(mode="json"),
        "instruction": "make this a radio",
    })

    from app.api import routes_classify

    seen = []

    class Capturing(OllamaClient):
        def generate_json(self, system, prompt):
            seen.append(system)
            return {"element": "radio", "title_lines": [0], "confidence": 0.9, "notes": "ok"}

        def status(self):
            return {"available": True, "reachable": True, "model_installed": True,
                    "url": "s", "model": "s", "installed_models": ["s"], "detail": "Ready."}

    monkeypatch.setattr(routes_classify, "build_client", Capturing)
    client.post("/api/quick-convert", json={"text": "Q1. Pick one\nYes\nNo"})

    assert "The survey programmer corrected an earlier question" in seen[0]


def test_quick_corrections_stay_out_of_the_document_flow(client, stub_command):
    """Phase 8's isolation still holds: separate libraries, no leakage."""
    from app.classify.corrections import correction_memory

    correction_memory.clear()
    client.post("/api/quick-command/confirm", json={
        "before": a_question().model_dump(mode="json"),
        "after": a_question(element="radio").model_dump(mode="json"),
        "instruction": "make this a radio",
    })

    assert correction_memory.recent() == [], "the document review's library is untouched"
    assert client.get("/api/corrections").json()["count"] == 0


# -- the existing editing path is unchanged --------------------------------


def test_structured_editing_needs_no_model_call(client, monkeypatch):
    """Checklist: the dropdown/field route still costs nothing."""
    from app.api import routes_classify

    def explode():
        raise AssertionError("structured editing must not call the model")

    monkeypatch.setattr(routes_classify, "build_client", explode)
    response = client.post("/api/quick-generate", json={
        "questions": [a_question(element="radio").model_dump(mode="json")]
    })

    assert response.status_code == 200
    assert '<radio label="Q1"' in response.json()["xml"]
