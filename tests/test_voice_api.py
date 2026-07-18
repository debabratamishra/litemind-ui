import asyncio
import importlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    import app.backend.api.voice as voice_mod

    calls = {}

    class FakeConn:
        pc_id = "test-pc"

        def __init__(self, ice_servers):
            self.ice_servers = ice_servers

        async def initialize(self, sdp, type):
            calls["init"] = (sdp, type)

        def event_handler(self, name):
            def deco(fn):
                return fn

            return deco

        def get_answer(self):
            return {"sdp": "fake-answer", "type": "answer", "pc_id": self.pc_id}

        async def renegotiate(self, sdp, type, restart_pc=False):
            calls["reneg"] = True

    monkeypatch.setattr(voice_mod, "SmallWebRTCConnection", FakeConn)
    monkeypatch.setattr(voice_mod, "run_voice_pipeline", lambda conn, settings: None)
    app = FastAPI()
    app.include_router(voice_mod.router)
    voice_mod.pcs_map.clear()
    yield TestClient(app)
    voice_mod.pcs_map.clear()


def test_offer_creates_connection(client):
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
    assert body["pc_id"] == "test-pc"
    assert body["type"] == "answer"
    import app.backend.api.voice as voice_mod

    assert "test-pc" in voice_mod.pcs_map


def test_offer_requires_pc_id(client):
    resp = client.post("/api/voice/offer", json={"sdp": "x", "type": "offer"})
    assert resp.status_code == 200
    assert resp.json().get("error")
