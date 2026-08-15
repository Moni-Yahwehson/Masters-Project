"""API contract tests — no model weights or running LLM required.

With the LLM unavailable the pipeline must still return a usable, honestly
labelled report rather than an error. These tests pin that behaviour.
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def _png_data_url(width: int = 64, height: int = 64) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (120, 90, 70)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def test_health(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["status"] in {"ok", "degraded"}
    assert body["knowledge_chunks"] > 0


def test_config(client: TestClient) -> None:
    body = client.get("/api/config").json()
    assert body["decision_threshold"] == pytest.approx(0.5)
    assert body["positive_class"] == "lumpycows"


def test_knowledge_documents(client: TestClient) -> None:
    documents = client.get("/api/knowledge/documents").json()
    assert len(documents) >= 7
    assert all(document["chunks"] > 0 for document in documents)


def test_knowledge_search(client: TestClient) -> None:
    passages = client.get(
        "/api/knowledge/search", params={"q": "vaccination coverage", "top_k": 3}
    ).json()
    assert passages
    assert passages[0]["score"] >= passages[-1]["score"]


def test_chat_requires_input(client: TestClient) -> None:
    assert client.post("/api/chat", json={"message": "", "images": []}).status_code == 400


def test_rejects_non_data_url(client: TestClient) -> None:
    response = client.post(
        "/api/diagnose/json",
        json={"message": "check", "images": [{"data_url": "https://example.com/a.jpg"}]},
    )
    assert response.status_code == 400


def test_rejects_too_many_images(client: TestClient) -> None:
    payload = {"data_url": _png_data_url(8, 8)}
    response = client.post(
        "/api/diagnose/json", json={"message": "x", "images": [payload] * 12}
    )
    assert response.status_code == 400
    assert "Too many images" in response.json()["detail"]


def test_diagnose_returns_a_report_and_trace(client: TestClient) -> None:
    response = client.post(
        "/api/diagnose/json",
        json={"message": "Diagnose this cow", "images": [{"data_url": _png_data_url()}]},
    )
    assert response.status_code == 200
    report = response.json()

    assert report["verdict"] in {
        "lsd_likely",
        "lsd_unlikely",
        "inconclusive",
        "not_cattle",
    }
    assert report["answer_markdown"], "a report must always carry a human-readable answer"
    assert report["trace"], "the agent trace must always be populated"
    # The standing safety framing must survive every degradation path.
    assert "not a diagnosis" in report["answer_markdown"].lower()
    agents = {step["agent"] for step in report["trace"]}
    assert "cnn_vision" in agents
    assert "knowledge_retrieval" in agents or report["triage"] is not None


def test_diagnose_stream_emits_steps_then_report(client: TestClient) -> None:
    with client.stream(
        "POST",
        "/api/diagnose/stream",
        json={"message": "Diagnose", "images": [{"data_url": _png_data_url()}]},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    assert "event: agent_step" in body
    assert "event: report" in body
    assert "event: done" in body
    assert body.index("event: agent_step") < body.index("event: report")


def test_session_round_trip(client: TestClient) -> None:
    first = client.post("/api/chat", json={"message": "How does LSD spread?"}).json()
    session_id = first["session_id"]
    assert first["markdown"]

    transcript = client.get(f"/api/chat/sessions/{session_id}").json()
    assert [message["role"] for message in transcript] == ["user", "assistant"]

    assert client.delete(f"/api/chat/sessions/{session_id}").json()["deleted"] is True
    assert client.get(f"/api/chat/sessions/{session_id}").status_code == 404
