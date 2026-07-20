"""Unit tests for ``app/backend/api/voice.py`` (WebRTC SDP offer endpoint).

The voice route is mounted in ``main.app``, so we test it there. The WebRTC
peer connection (``SmallWebRTCConnection``) and the Pipecat pipeline runner
(``run_voice_pipeline``) are mocked at their boundaries, so no real WebRTC /
Pipecat / network activity occurs offline.
"""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.backend.api import voice as voice_api
from main import app


@pytest.fixture
def voice_client(monkeypatch):
    calls = {}

    class FakeConn:
        pc_id = "test-pc"

        def __init__(self, ice_servers):
            self.ice_servers = ice_servers

        async def initialize(self, sdp, type):
            calls["init"] = (sdp, type)

        def event_handler(self, name):
            # The route decorates a handler for the "closed" event; we just
            # return an identity decorator and never invoke it.
            def decorator(fn):
                return fn

            return decorator

        def get_answer(self):
            return {"sdp": "fake-answer", "type": "answer", "pc_id": self.pc_id}

        async def renegotiate(self, sdp, type, restart_pc=False):
            calls["renegotiate"] = (sdp, type, restart_pc)

    monkeypatch.setattr(voice_api, "SmallWebRTCConnection", FakeConn)
    # Background task must not touch a real Pipecat pipeline.
    monkeypatch.setattr(voice_api, "run_voice_pipeline", AsyncMock())

    voice_api.pcs_map.clear()
    client = TestClient(app)
    yield client, calls
    voice_api.pcs_map.clear()


def test_voice_offer_creates_connection_and_returns_answer(voice_client):
    client, calls = voice_client

    resp = client.post(
        "/api/voice/offer",
        json={
            "pc_id": "test-pc",
            "sdp": "v=0...",
            "type": "offer",
            "model": "llama3",
            "backend": "ollama",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "answer"
    assert body["sdp"] == "fake-answer"
    assert body["pc_id"] == "test-pc"

    # Connection was initialised with the supplied offer and registered.
    assert calls.get("init") == ("v=0...", "offer")
    assert "test-pc" in voice_api.pcs_map


def test_voice_offer_requires_pc_id(voice_client):
    client, _ = voice_client

    resp = client.post("/api/voice/offer", json={"sdp": "x", "type": "offer"})
    assert resp.status_code == 200
    assert resp.json().get("error") == "pc_id required"


def test_voice_offer_renegotiates_existing_connection(voice_client):
    client, calls = voice_client

    first = client.post(
        "/api/voice/offer",
        json={"pc_id": "test-pc", "sdp": "v=0...", "type": "offer"},
    )
    assert first.status_code == 200
    calls.clear()

    second = client.post(
        "/api/voice/offer",
        json={"pc_id": "test-pc", "sdp": "v=0...", "type": "offer", "restart_pc": True},
    )
    assert second.status_code == 200
    # Reuse path: renegotiate instead of initialize.
    assert "renegotiate" in calls
    assert "init" not in calls


def test_voice_offer_runs_pipeline_as_background_task(voice_client, monkeypatch):
    # ``run_voice_pipeline_safe`` is the background task; it delegates to the
    # mocked ``run_voice_pipeline``. We confirm the task runs to completion.
    ran = {}

    async def _fake_pipeline(conn, settings):
        ran["ok"] = True

    monkeypatch.setattr(voice_api, "run_voice_pipeline", _fake_pipeline)

    resp = client_post_offer(voice_client)
    assert resp.status_code == 200
    assert ran.get("ok") is True


def client_post_offer(voice_client):
    client, _ = voice_client
    return client.post(
        "/api/voice/offer",
        json={"pc_id": "test-pc", "sdp": "v=0...", "type": "offer"},
    )


def test_voice_offer_pipeline_failure_is_swallowed(voice_client, monkeypatch):
    """A pipeline failure must not surface as an HTTP error to the client."""

    async def _boom(conn, settings):
        raise RuntimeError("pipeline down")

    monkeypatch.setattr(voice_api, "run_voice_pipeline", _boom)

    client, _ = voice_client
    resp = client.post(
        "/api/voice/offer",
        json={"pc_id": "test-pc", "sdp": "v=0...", "type": "offer"},
    )
    # The answer is still returned; the failure is handled inside the task.
    assert resp.status_code == 200
    assert resp.json()["type"] == "answer"
