import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.backend.api.deps import User, get_system_user


@pytest.fixture
def client():
    import backend.app.backend.api.voice as voice_mod

    app = FastAPI()
    app.include_router(voice_mod.router)
    # Voice offer resolves the seeded system user; provide a user for the tests.
    app.dependency_overrides[get_system_user] = lambda: User(id="u1", email="u1@x.com")
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
    import backend.app.backend.api.voice as voice_mod

    assert "test-pc" in voice_mod.pcs_map


def test_offer_requires_pc_id(client):
    resp = client.post("/api/voice/offer", json={"sdp": "x", "type": "offer"})
    assert resp.status_code == 200
    assert resp.json().get("error")
