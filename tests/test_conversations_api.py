"""
Tests for the conversation CRUD API (backend/app/backend/api/conversations.py).

Uses an in-memory FakeStore and dependency overrides so no live PostgreSQL is
needed. Every request resolves the seeded system user (see api/deps.py), so
cross-user isolation no longer applies.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.backend.api.conversations import get_store, router
from tests.test_conversation_store import FakeStore


def _make_app():
    """Build a minimal app with the conversations router."""
    app = FastAPI()
    store = FakeStore()
    app.include_router(router)
    app.dependency_overrides[get_store] = lambda: store
    return app, store


def _client():
    app, store = _make_app()
    return TestClient(app), store


def test_create_and_get():
    client, _ = _client()
    r = client.post("/api/conversations", json={"title": "My chat"})
    assert r.status_code == 200
    cid = r.json()["id"]
    got = client.get(f"/api/conversations/{cid}")
    assert got.status_code == 200 and got.json()["title"] == "My chat"


def test_messages_flow():
    client, _ = _client()
    cid = client.post("/api/conversations", json={}).json()["id"]
    add = client.post(
        f"/api/conversations/{cid}/messages",
        json={"role": "user", "content": "hi"},
    )
    assert add.status_code == 200
    msgs = client.get(f"/api/conversations/{cid}/messages").json()["messages"]
    assert len(msgs) == 1 and msgs[0]["content"] == "hi"


def test_update_and_delete():
    client, _ = _client()
    cid = client.post("/api/conversations", json={"title": "old"}).json()["id"]
    upd = client.patch(f"/api/conversations/{cid}", json={"title": "new"})
    assert upd.status_code == 200 and upd.json()["title"] == "new"
    assert client.delete(f"/api/conversations/{cid}").status_code == 200
    assert client.get(f"/api/conversations/{cid}").status_code == 404
