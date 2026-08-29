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


@pytest.fixture(autouse=True)
def isolated_correction_library(monkeypatch):
    """Keep every test's corrections in memory and to itself.

    Without this the suite would read and write the developer's real
    corrections file, and one test's entries would rehydrate into the next.
    """
    from app.classify import corrections as corrections_module
    from app.classify.library import CorrectionLibrary

    library = CorrectionLibrary(None)
    monkeypatch.setattr(corrections_module.correction_memory, "_library", library)
    corrections_module.correction_memory.clear()

    try:
        from app.api import routes_quick

        monkeypatch.setattr(routes_quick.quick_corrections, "_library", library)
        routes_quick.quick_corrections.clear()
        routes_quick.quick_corrections.use_document("quick-convert")
    except ImportError:  # pragma: no cover
        pass

    yield

    corrections_module.correction_memory.clear()
