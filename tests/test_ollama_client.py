"""The Ollama client's request shape and its failure reporting.

Ollama answers /api/generate with 404 when the *model* is unknown, not only
when the URL is. Reporting that as a bare 404 sent a real user hunting for an
endpoint problem for hours, so these tests pin both the request the app builds
and the message it gives back when the request fails.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from app.classify.ollama import OllamaClient, OllamaError, OllamaModelMissing, _tag_key


class FakeOllama(BaseHTTPRequestHandler):
    """Behaves like a real Ollama: unknown model -> 404 with an explanation."""

    installed = ["llama3:latest"]
    received: list[dict] = []
    headers_seen: list[dict] = []

    def log_message(self, *args):
        pass

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/tags"):
            self._send(200, {"models": [{"name": n, "model": n} for n in self.installed]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).received.append(payload)
        type(self).headers_seen.append(dict(self.headers))

        known = {n.split(":")[0] for n in self.installed} | set(self.installed)
        if payload.get("model") not in known:
            self._send(404, {
                "error": f"model '{payload.get('model')}' not found, try pulling it first"
            })
            return
        self._send(200, {"response": json.dumps({"element": "radio", "confidence": 0.9})})


@pytest.fixture
def server():
    FakeOllama.received = []
    FakeOllama.headers_seen = []
    FakeOllama.installed = ["llama3:latest"]

    httpd = HTTPServer(("127.0.0.1", 0), FakeOllama)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


# -- the request the app builds -------------------------------------------


def test_the_request_url_is_exactly_api_generate(server):
    OllamaClient(base_url=server, model="llama3").generate_json("sys", "prompt")
    assert FakeOllama.received, "the request reached /api/generate"


def test_a_trailing_slash_on_the_base_url_does_not_double_up(server):
    OllamaClient(base_url=server + "/", model="llama3").generate_json("sys", "prompt")
    assert len(FakeOllama.received) == 1


def test_the_request_body_has_the_shape_ollama_expects(server):
    OllamaClient(base_url=server, model="llama3").generate_json("SYSTEM", "PROMPT")
    (payload,) = FakeOllama.received

    assert payload["model"] == "llama3"
    assert payload["system"] == "SYSTEM"
    assert payload["prompt"] == "PROMPT"
    assert payload["stream"] is False
    assert payload["format"] == "json"
    assert payload["options"] == {"temperature": 0}


def test_the_request_is_sent_as_json(server):
    OllamaClient(base_url=server, model="llama3").generate_json("sys", "prompt")
    (headers,) = FakeOllama.headers_seen
    assert headers["Content-Type"] == "application/json"


def test_the_configured_model_name_is_sent_verbatim(server):
    """Including a tag — the client must not rewrite what was configured."""
    FakeOllama.installed = ["mistral:7b"]
    OllamaClient(base_url=server, model="mistral:7b").generate_json("s", "p")
    assert FakeOllama.received[0]["model"] == "mistral:7b"


# -- the failure that looked like a URL problem ---------------------------


def test_a_missing_model_says_so_instead_of_reporting_a_bare_404(server):
    client = OllamaClient(base_url=server, model="llama3.1")

    with pytest.raises(OllamaModelMissing) as caught:
        client.generate_json("sys", "prompt")

    message = str(caught.value)
    assert "'llama3.1' is not installed" in message
    assert "llama3:latest" in message, "the message names what IS installed"
    assert "ollama pull llama3.1" in message
    assert "DECIPHER_OLLAMA_MODEL" in message
    assert "model 'llama3.1' not found" in message, "Ollama's own words are kept"


def test_a_missing_model_is_still_an_ollama_error(server):
    """The fallback path catches OllamaError, so classification must degrade."""
    client = OllamaClient(base_url=server, model="llama3.1")
    with pytest.raises(OllamaError):
        client.generate_json("sys", "prompt")


def test_other_http_errors_keep_their_status_and_body(server):
    FakeOllama.installed = []
    client = OllamaClient(base_url=server, model="llama3.1")

    with pytest.raises(OllamaError) as caught:
        client.generate_json("sys", "prompt")
    assert "not installed" in str(caught.value)


def test_an_unreachable_runtime_names_the_url():
    client = OllamaClient(base_url="http://127.0.0.1:1", model="llama3", timeout=2.0)

    with pytest.raises(OllamaError) as caught:
        client.generate_json("sys", "prompt")
    assert "http://127.0.0.1:1" in str(caught.value)


# -- availability must mean "ready", not "answering" ----------------------


def test_reachable_but_missing_the_model_is_not_available(server):
    """The old check reported "AI ready" while every call 404'd."""
    client = OllamaClient(base_url=server, model="llama3.1")

    assert client.is_reachable() is True
    assert client.has_model() is False
    assert client.is_available() is False


def test_an_installed_model_is_available(server):
    assert OllamaClient(base_url=server, model="llama3").is_available() is True


@pytest.mark.parametrize(
    "configured,expected",
    [("llama3", True), ("llama3:latest", True), ("llama3.1", False), ("mistral", False)],
)
def test_a_bare_name_matches_its_latest_tag(server, configured, expected):
    """Ollama resolves llama3 to llama3:latest, so the check must too."""
    assert OllamaClient(base_url=server, model=configured).has_model() is expected


@pytest.mark.parametrize(
    "left,right",
    [("llama3", "llama3:latest"), ("llama3:latest", "llama3"), ("a:b", "a:b")],
)
def test_tag_normalisation(left, right):
    assert _tag_key(left) == _tag_key(right)


def test_status_explains_a_missing_model(server):
    status = OllamaClient(base_url=server, model="llama3.1").status()

    assert status["available"] is False
    assert status["reachable"] is True
    assert status["model_installed"] is False
    assert status["installed_models"] == ["llama3:latest"]
    assert "not installed" in status["detail"]


def test_status_explains_an_unreachable_runtime():
    status = OllamaClient(base_url="http://127.0.0.1:1", model="llama3").status()

    assert status["available"] is False
    assert status["reachable"] is False
    assert "not reachable" in status["detail"]


def test_status_is_positive_when_ready(server):
    status = OllamaClient(base_url=server, model="llama3").status()

    assert status["available"] is True
    assert status["detail"] == "Ready with llama3."


def test_no_models_installed_is_reported_clearly(server):
    FakeOllama.installed = []
    status = OllamaClient(base_url=server, model="llama3").status()

    assert status["available"] is False
    assert "no models installed" in status["detail"]
