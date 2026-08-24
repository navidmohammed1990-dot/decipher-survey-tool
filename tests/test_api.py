"""HTTP surface of the Phase 1 app."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def _upload(client, name, payload):
    return client.post(
        "/api/parse",
        files={"file": (name, payload,
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    )


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_index_serves_the_review_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Decipher Survey Base Generator" in response.text


def test_parse_returns_the_document_model(client, sample_docx):
    response = _upload(client, "sample.docx", sample_docx.read_bytes())
    assert response.status_code == 200

    payload = response.json()
    assert payload["source_filename"] == "sample.docx"
    assert [q["label"] for q in payload["questions"] if not q["is_preamble"]] == [
        "S1", "Q5", "Q6", "Q7",
    ]
    assert payload["stats"]["tables"] == 1


def test_parse_preserves_formatting_in_the_response(client, sample_docx):
    payload = _upload(client, "sample.docx", sample_docx.read_bytes()).json()
    q5 = next(q for q in payload["questions"] if q["label"] == "Q5")

    assert [r["text"] for r in q5["title_runs"] if r["bold"]] == ["purchased"]
    assert [r["text"] for r in q5["title_runs"] if r["italic"]] == ["6 months"]


def test_rejects_wrong_extension(client):
    response = client.post("/api/parse", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 415
    assert "Unsupported file type" in response.json()["detail"]


def test_rejects_empty_upload(client):
    response = _upload(client, "empty.docx", b"")
    assert response.status_code == 400


def test_rejects_a_file_that_is_not_a_real_docx(client):
    response = _upload(client, "fake.docx", b"definitely not a zip archive")
    assert response.status_code == 422
    assert "could not be read" in response.json()["detail"]


def test_rejects_an_oversized_upload(client, monkeypatch):
    from dataclasses import replace

    from app.api import routes_parse

    monkeypatch.setattr(
        routes_parse, "settings", replace(routes_parse.settings, max_upload_bytes=10)
    )
    response = _upload(client, "big.docx", b"x" * 100)

    assert response.status_code == 413
    assert "upload limit" in response.json()["detail"]


def test_openapi_documents_the_parse_endpoint(client):
    schema = client.get("/openapi.json").json()
    assert "/api/parse" in schema["paths"]
