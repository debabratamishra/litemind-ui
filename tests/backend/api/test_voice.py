"""Unit tests for ``backend/app/backend/api/voice.py`` (ElevenLabs ConvAI session)."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.app.backend.api.deps import User, get_system_user
from backend.main import app

app.dependency_overrides[get_system_user] = lambda: User(id="u1", email="u1@x.com")


@pytest.fixture
def client():
    return TestClient(app)


def test_create_voice_session_success(client):
    with (
        patch(
            "backend.app.backend.api.voice.resolve_elevenlabs_credentials",
            new=AsyncMock(return_value=("fake-key", "agent-123")),
        ),
        patch(
            "backend.app.backend.api.voice.get_convai_signed_url",
            new=AsyncMock(return_value="wss://api.elevenlabs.io/v1/convai/conversation?signed=true"),
        ),
    ):
        resp = client.post("/api/voice/session", json={"agent_id": "agent-123"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["signed_url"] == "wss://api.elevenlabs.io/v1/convai/conversation?signed=true"
        assert data["agent_id"] == "agent-123"


def test_create_voice_session_missing_key(client):
    from backend.app.services.voice_pipeline import VoiceConfigurationError

    with patch(
        "backend.app.backend.api.voice.resolve_elevenlabs_credentials",
        side_effect=VoiceConfigurationError("ElevenLabs API key is required"),
    ):
        resp = client.post("/api/voice/session", json={})
        assert resp.status_code == 400
        assert "ElevenLabs API key is required" in resp.json()["detail"]


def test_get_voice_session(client):
    with (
        patch(
            "backend.app.backend.api.voice.resolve_elevenlabs_credentials",
            new=AsyncMock(return_value=("fake-key", "agent-123")),
        ),
        patch(
            "backend.app.backend.api.voice.get_convai_signed_url",
            new=AsyncMock(return_value="wss://signed-url"),
        ),
    ):
        resp = client.get("/api/voice/session?agent_id=agent-123")
        assert resp.status_code == 200
        assert resp.json()["signed_url"] == "wss://signed-url"


def test_offer_deprecated(client):
    resp = client.post("/api/voice/offer", json={"pc_id": "123"})
    assert resp.status_code == 200
    assert "deprecated" in resp.json()["error"].lower() or "elevenlabs" in resp.json()["error"].lower()
