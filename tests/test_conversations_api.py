"""
Tests for the conversation CRUD API (app/backend/api/conversations.py).

Verifies authentication is required and that every operation is isolated to
the authenticated user. Uses an in-memory FakeStore and dependency overrides
so no live PostgreSQL or GoTrue is needed.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.backend.api.auth_deps import User, get_current_user
from app.backend.api.conversations import get_store, router
from tests.test_conversation_store import FakeStore


def _make_app(user_id: str | None):
    """Build a minimal app with the conversations router.

    If ``user_id`` is None, no auth override is installed (so requests are
    unauthenticated and should 401).
    """
    app = FastAPI()
    store = FakeStore()
    app.include_router(router)
    app.dependency_overrides[get_store] = lambda: store
    if user_id is not None:
        app.dependency_overrides[get_current_user] = lambda: User(id=user_id, email=f"{user_id}@x.com")
    return app, store


def _client(user_id: str | None):
    app, store = _make_app(user_id)
    return TestClient(app), store


def test_requires_auth():
    client, _ = _client(None)
    assert client.get("/api/conversations").status_code == 401


def test_create_and_get():
    client, _ = _client("u1")
    r = client.post("/api/conversations", json={"title": "My chat"})
    assert r.status_code == 200
    cid = r.json()["id"]
    got = client.get(f"/api/conversations/{cid}")
    assert got.status_code == 200 and got.json()["title"] == "My chat"


def test_isolation_between_users():
    client_a, store_a = _client("u1")
    created = client_a.post("/api/conversations", json={"title": "A"})
    cid = created.json()["id"]

    # User B cannot fetch A's conversation.
    client_b, _ = _client("u2")
    assert client_b.get(f"/api/conversations/{cid}").status_code == 404
    assert client_b.delete(f"/api/conversations/{cid}").status_code == 404
    # B's list is empty; A's list has the one conversation.
    assert client_b.get("/api/conversations").json()["conversations"] == []
    assert len(client_a.get("/api/conversations").json()["conversations"]) == 1


def test_messages_flow_and_isolation():
    client_a, _ = _client("u1")
    cid = client_a.post("/api/conversations", json={}).json()["id"]
    add = client_a.post(
        f"/api/conversations/{cid}/messages",
        json={"role": "user", "content": "hi"},
    )
    assert add.status_code == 200
    msgs = client_a.get(f"/api/conversations/{cid}/messages").json()["messages"]
    assert len(msgs) == 1 and msgs[0]["content"] == "hi"

    client_b, _ = _client("u2")
    # B cannot append to A's conversation (ownership enforced in the store).
    assert client_b.post(
        f"/api/conversations/{cid}/messages",
        json={"role": "user", "content": "intrude"},
    ).status_code == 404


def test_update_and_delete():
    client_a, _ = _client("u1")
    cid = client_a.post("/api/conversations", json={"title": "old"}).json()["id"]
    upd = client_a.patch(f"/api/conversations/{cid}", json={"title": "new"})
    assert upd.status_code == 200 and upd.json()["title"] == "new"
    assert client_a.delete(f"/api/conversations/{cid}").status_code == 200
    assert client_a.get(f"/api/conversations/{cid}").status_code == 404
