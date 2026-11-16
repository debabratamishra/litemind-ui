"""Error-response safety tests for the TTS router."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.backend.api import tts as tts_api


@pytest.fixture
def tts_client(monkeypatch):
    service = MagicMock()
    service.is_available.return_value = True
    service.synthesize = AsyncMock(side_effect=RuntimeError("tts engine missing at /opt/models/tts"))
    service.synthesize_text_chunk = MagicMock(side_effect=RuntimeError("tts engine missing at /opt/models/tts"))
    service.get_status.return_value = {"available": True}

    monkeypatch.setattr(tts_api, "get_tts_service", lambda: service)

    app = FastAPI()
    app.include_router(tts_api.router)
    return TestClient(app), service


def test_synthesize_error_is_generic(tts_client):
    client, _ = tts_client

    resp = client.post("/api/tts/synthesize", json={"text": "hello"})

    assert resp.status_code == 500
    assert "/opt/models/tts" not in resp.text


def test_synthesize_chunk_error_is_generic(tts_client):
    client, _ = tts_client

    resp = client.post("/api/tts/synthesize-chunk", json={"text": "hello"})

    assert resp.status_code == 500
    assert "/opt/models/tts" not in resp.text
