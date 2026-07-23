"""
Auth-gate tests for the chat endpoints (app/backend/api/chat.py).

Verifies that the chat, stream, web-search, and memory endpoints require a
valid user (401 when unauthenticated) and accept requests once a user is
injected via dependency override. LLM calls are stubbed so no model is needed.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.backend.api import chat as chat_api
from app.backend.api.auth_deps import User, get_current_user


def _make_app(authed: bool):
    app = FastAPI()
    app.include_router(chat_api.router)
    if authed:
        app.dependency_overrides[get_current_user] = lambda: User(id="u1", email="u1@x.com")
    return app


async def _fake_complete(*args, **kwargs):
    # complete_text returns a plain string (the production code wraps it in ChatResponse)
    return "stub-response"


async def _fake_stream(*args, **kwargs):
    yield "stub"
    yield "-chunk"


def test_stream_requires_auth():
    client = TestClient(_make_app(authed=False))
    r = client.post("/api/chat/stream", json={"message": "hi"})
    assert r.status_code == 401


def test_chat_requires_auth():
    client = TestClient(_make_app(authed=False))
    r = client.post("/api/chat", json={"message": "hi"})
    assert r.status_code == 401


def test_memory_stats_requires_auth():
    client = TestClient(_make_app(authed=False))
    r = client.get("/api/chat/memory/stats/abc")
    assert r.status_code == 401


def test_web_search_requires_auth():
    client = TestClient(_make_app(authed=False))
    r = client.post("/api/chat/web-search", json={"message": "hi", "use_web_search": True})
    assert r.status_code == 401


def test_chat_with_auth_uses_stub(monkeypatch):
    monkeypatch.setattr(chat_api, "complete_text", _fake_complete)
    client = TestClient(_make_app(authed=True))
    r = client.post("/api/chat", json={"message": "hi"})
    assert r.status_code == 200
    assert r.json()["response"] == "stub-response"


def test_stream_with_auth_uses_stub(monkeypatch):
    monkeypatch.setattr(chat_api, "stream_completion", _fake_stream)
    client = TestClient(_make_app(authed=True))
    r = client.post("/api/chat/stream", json={"message": "hi"})
    assert r.status_code == 200
    body = r.text
    assert "stub" in body and "chunk" in body
