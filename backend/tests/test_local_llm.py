"""Local LLM client tests, run against a stub inference server.

Ollama is not assumed to be installed, so these tests stand up a tiny HTTP
server that mimics its `/api/tags` and `/api/chat` endpoints. That exercises the
real client code path — request shape, image attachment, schema-constrained
decoding, JSON recovery, the repair retry, and the "server is down" degradation
— without needing multi-gigabyte model weights in CI.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Optional

import pytest
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.llm import (  # noqa: E402
    ImagePayload,
    LLMUnavailable,
    LocalLLMClient,
    _extract_json,
    image_blocks,
)


class Sample(BaseModel):
    verdict: str
    confidence: float


class _StubOllama(BaseHTTPRequestHandler):
    """Mimics the subset of the Ollama API the client uses."""

    #: Queue of reply bodies; each /api/chat call pops the next one.
    replies: list[str] = []
    #: Every received /api/chat request body, for assertions.
    requests: list[dict[str, Any]] = []
    models: list[str] = ["llava:7b", "llama3.1:8b"]

    def log_message(self, *args: Any) -> None:  # silence the test output
        pass

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        if self.path == "/api/tags":
            self._send(200, {"models": [{"name": name} for name in self.models]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).requests.append(body)
        if self.path != "/api/chat":
            self._send(404, {"error": "not found"})
            return
        reply = (
            type(self).replies.pop(0)
            if type(self).replies
            else json.dumps({"verdict": "ok", "confidence": 0.5})
        )
        self._send(200, {"message": {"role": "assistant", "content": reply}})


@pytest.fixture()
def stub_server(monkeypatch: pytest.MonkeyPatch):
    """Run the stub on a free port and point the client at it."""
    server = HTTPServer(("127.0.0.1", 0), _StubOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    _StubOllama.replies = []
    _StubOllama.requests = []
    _StubOllama.models = ["llava:7b", "llama3.1:8b"]

    settings = get_settings()
    monkeypatch.setattr(
        settings, "llm_base_url", f"http://127.0.0.1:{server.server_port}"
    )
    monkeypatch.setattr(settings, "vision_model", "llava:7b")
    monkeypatch.setattr(settings, "text_model", "llama3.1:8b")
    monkeypatch.setattr(settings, "enable_llm", True)

    try:
        yield _StubOllama
    finally:
        server.shutdown()
        server.server_close()


def _client() -> LocalLLMClient:
    return LocalLLMClient()


# -- availability ----------------------------------------------------------


def test_available_when_server_and_models_present(stub_server) -> None:
    assert _client().available


def test_unavailable_when_server_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    # Port 1 is reserved and never listening.
    monkeypatch.setattr(settings, "llm_base_url", "http://127.0.0.1:1")
    client = _client()
    assert not client.available
    assert "Cannot reach the local model server" in client.unavailable_reason


def test_unavailable_when_model_not_pulled(stub_server) -> None:
    stub_server.models = ["llava:7b"]  # text model missing
    client = _client()
    assert not client.available
    assert "not installed locally" in client.unavailable_reason
    assert "ollama pull" in client.unavailable_reason


def test_bare_model_name_matches_tagged_install(stub_server, monkeypatch) -> None:
    """`llava` in config should match an installed `llava:7b`."""
    monkeypatch.setattr(get_settings(), "vision_model", "llava")
    assert _client().available


def test_disabled_by_configuration(stub_server, monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "enable_llm", False)
    client = _client()
    assert not client.available
    assert "disabled" in client.unavailable_reason.lower()


# -- request shape ---------------------------------------------------------


def test_text_request_shape(stub_server) -> None:
    stub_server.replies = ["a plain answer"]
    answer = _client().text(
        system="be brief", content=[{"type": "text", "text": "hello"}]
    )
    assert answer == "a plain answer"

    sent = stub_server.requests[-1]
    assert sent["model"] == "llama3.1:8b"
    assert sent["stream"] is False
    assert sent["messages"][0] == {"role": "system", "content": "be brief"}
    assert sent["messages"][1]["content"] == "hello"
    assert "images" not in sent["messages"][1]
    assert sent["options"]["temperature"] == pytest.approx(0.1)


def test_images_are_attached_to_the_user_message(stub_server) -> None:
    stub_server.replies = [json.dumps({"verdict": "lsd_likely", "confidence": 0.8})]
    images = [
        ImagePayload.from_bytes(b"\x89PNG-one", "img-1", "image/png"),
        ImagePayload.from_bytes(b"\x89PNG-two", "img-2", "image/png"),
    ]
    content = [{"type": "text", "text": "look at these"}] + image_blocks(images)

    _client().structured(
        system="observe", content=content, output_model=Sample, model="llava:7b"
    )

    sent = stub_server.requests[-1]
    assert sent["model"] == "llava:7b"
    assert len(sent["messages"][1]["images"]) == 2
    assert sent["messages"][1]["images"][0] == images[0].base64_data
    # Image ids stay in the text so the model can refer to them.
    assert "img-1" in sent["messages"][1]["content"]


def test_keep_alive_is_sent(stub_server) -> None:
    """Without keep_alive Ollama evicts the model between agents, and each of
    the pipeline's four sequential calls pays the full load cost again."""
    stub_server.replies = ["ok"]
    _client().text(system="s", content=[{"type": "text", "text": "q"}])
    assert stub_server.requests[-1]["keep_alive"] == get_settings().llm_keep_alive


def test_both_roles_default_to_one_model() -> None:
    """One multimodal model for both roles is what keeps this usable on
    consumer hardware — Ollama holds one model in memory at a time, so a split
    means evicting and reloading gigabytes between agents."""
    settings = get_settings()
    assert settings.vision_model == settings.text_model, (
        "vision and text models differ, which reintroduces model swapping"
    )


def test_structured_sends_a_json_schema(stub_server) -> None:
    stub_server.replies = [json.dumps({"verdict": "ok", "confidence": 0.1})]
    _client().structured(
        system="s", content=[{"type": "text", "text": "q"}], output_model=Sample
    )
    schema = stub_server.requests[-1]["format"]
    assert schema["type"] == "object"
    assert set(schema["properties"]) == {"verdict", "confidence"}


# -- parsing and recovery --------------------------------------------------


def test_structured_parses_valid_json(stub_server) -> None:
    stub_server.replies = [json.dumps({"verdict": "lsd_unlikely", "confidence": 0.25})]
    result = _client().structured(
        system="s", content=[{"type": "text", "text": "q"}], output_model=Sample
    )
    assert result.verdict == "lsd_unlikely"
    assert result.confidence == pytest.approx(0.25)


@pytest.mark.parametrize(
    "wrapped",
    [
        '```json\n{"verdict": "ok", "confidence": 0.4}\n```',
        'Sure! Here you go:\n{"verdict": "ok", "confidence": 0.4}\nHope that helps.',
        '```\n{"verdict": "ok", "confidence": 0.4}\n```',
    ],
)
def test_structured_recovers_json_from_prose(stub_server, wrapped: str) -> None:
    """A model that ignores the grammar must not break the pipeline."""
    stub_server.replies = [wrapped]
    result = _client().structured(
        system="s", content=[{"type": "text", "text": "q"}], output_model=Sample
    )
    assert result.confidence == pytest.approx(0.4)


def test_structured_repairs_after_an_invalid_first_reply(stub_server) -> None:
    stub_server.replies = [
        '{"verdict": "ok"}',  # missing a required field
        '{"verdict": "ok", "confidence": 0.7}',
    ]
    result = _client().structured(
        system="s", content=[{"type": "text", "text": "q"}], output_model=Sample
    )
    assert result.confidence == pytest.approx(0.7)
    assert len(stub_server.requests) == 2
    # The repair turn shows the model its own bad output.
    assert "did not satisfy the schema" in stub_server.requests[1]["messages"][1]["content"]


def test_structured_gives_up_after_one_repair(stub_server) -> None:
    stub_server.replies = ['{"nope": 1}', '{"still": "wrong"}']
    with pytest.raises(LLMUnavailable):
        _client().structured(
            system="s", content=[{"type": "text", "text": "q"}], output_model=Sample
        )


def test_empty_response_is_an_error(stub_server) -> None:
    stub_server.replies = ["   "]
    with pytest.raises(LLMUnavailable):
        _client().text(system="s", content=[{"type": "text", "text": "q"}])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"a": 1}', '{"a": 1}'),
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ('prefix {"a": 1} suffix', '{"a": 1}'),
        ("no json here", "no json here"),
    ],
)
def test_extract_json(raw: str, expected: str) -> None:
    assert _extract_json(raw) == expected


# -- image payloads --------------------------------------------------------


def test_data_url_round_trip() -> None:
    payload = ImagePayload.from_bytes(b"hello", "img-1", "image/png")
    url = f"data:image/png;base64,{payload.base64_data}"
    decoded = ImagePayload.from_data_url(url, "img-2")
    assert decoded.to_bytes() == b"hello"
    assert decoded.media_type == "image/png"


@pytest.mark.parametrize(
    "bad", ["https://example.com/a.png", "data:image/png;base64,!!!not-base64!!!", ""]
)
def test_bad_data_urls_raise(bad: str) -> None:
    with pytest.raises(ValueError):
        ImagePayload.from_data_url(bad, "img-1")
