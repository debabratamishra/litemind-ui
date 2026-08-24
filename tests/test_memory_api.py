"""API tests for /api/memory — auth required, user-scoped CRUD."""

import uuid

import pytest
from fastapi.testclient import TestClient


class FakeStore:
    def __init__(self):
        self.rows = {}

    async def list_memories(self, user_id, limit=50):
        result = [r for r in self.rows.values() if r.user_id == user_id]
        return result

    async def add_memory(self, user_id, content, source="auto"):
        from app.backend.user_memory_store import UserMemoryRecord

        rec = UserMemoryRecord(
            id=str(uuid.uuid4()), user_id=user_id, content=content, source=source,
            created_at="2026-08-23T00:00:00", updated_at="2026-08-23T00:00:00",
        )
        self.rows[rec.id] = rec
        return rec

    async def update_memory(self, user_id, memory_id, content):
        r = self.rows.get(memory_id)
        if not r or r.user_id != user_id:
            return False
        r.content = content
        return True

    async def delete_memory(self, user_id, memory_id):
        r = self.rows.get(memory_id)
        if not r or r.user_id != user_id:
            return False
        del self.rows[memory_id]
        return True

    async def clear_memories(self, user_id):
        ids = [i for i, r in self.rows.items() if r.user_id == user_id]
        for i in ids:
            del self.rows[i]
        return len(ids)


@pytest.fixture()
def client(monkeypatch):
    from app.backend.api.auth_deps import User, get_current_user
    from main import app

    store = FakeStore()
    # Patch the function where it's defined to affect all importers
    def get_store():
        return store
    monkeypatch.setattr("app.backend.user_memory_store.get_user_memory_store", get_store)
    # Same override pattern as tests/test_chat_auth.py:_make_app
    # Create a single user instance to return consistently
    test_user = User(id=str(uuid.uuid4()), email="u1@x.com")
    app.dependency_overrides[get_current_user] = lambda: test_user
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.pop(get_current_user, None)


def test_memory_requires_auth():
    # Fresh app instead of main.app: other test modules install module-level
    # dependency_overrides on the shared main.app that never clean up, which
    # would authenticate this request and turn the 401 into a store error.
    # Router mounting on main.app is covered by the fixture-based tests below.
    from fastapi import FastAPI

    from app.backend.api.memory import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    assert client.get("/api/memory").status_code == 401


def test_memory_crud_roundtrip(client):
    resp = client.post("/api/memory", json={"content": "Prefers dark mode"})
    assert resp.status_code == 200
    mid = resp.json()["id"]
    assert resp.json()["source"] == "manual"

    listed = client.get("/api/memory").json()
    assert [m["content"] for m in listed] == ["Prefers dark mode"]

    assert client.put(f"/api/memory/{mid}", json={"content": "Prefers light mode"}).status_code == 200
    assert client.get("/api/memory").json()[0]["content"] == "Prefers light mode"

    assert client.delete(f"/api/memory/{mid}").status_code == 200
    assert client.get("/api/memory").json() == []


def test_memory_add_rejects_blank_content(client):
    assert client.post("/api/memory", json={"content": "   "}).status_code == 422


def test_memory_add_rejects_overlong_content(client):
    assert client.post("/api/memory", json={"content": "x" * 501}).status_code == 422


def test_memory_update_unknown_id_404(client):
    assert client.put(f"/api/memory/{uuid.uuid4()}", json={"content": "x"}).status_code == 404


def test_memory_clear(client):
    client.post("/api/memory", json={"content": "a"})
    client.post("/api/memory", json={"content": "b"})
    resp = client.post("/api/memory/clear")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == 2
    assert client.get("/api/memory").json() == []
