from __future__ import annotations

import pytest

from tests.fixtures.build_fixture import build


@pytest.fixture(scope="session")
def sample_docx(tmp_path_factory):
    """The generated sample questionnaire, built once per test session."""
    return build(tmp_path_factory.mktemp("fixtures") / "sample_questionnaire.docx")


@pytest.fixture(scope="session")
def parsed_sample(sample_docx):
    from app.parsing.docx_parser import parse_docx

    return parse_docx(sample_docx)
