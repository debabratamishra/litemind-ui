"""Auth removal: a route that previously required auth no longer returns 401.

``GET /api/conversations`` used to answer 401 without a Bearer token or
session cookie. After the GoTrue strip, every route resolves the seeded
system user, so the same request must not be an auth failure.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.backend.api.conversations import get_store, router
from tests.test_conversation_store import FakeStore


def _client() -> TestClient:
    # Local app + FakeStore, mirroring the fixture pattern in tests/backend/api/.
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_store] = lambda: FakeStore()
    return TestClient(app)


def test_conversations_without_token_is_not_unauthorized():
    client = _client()
    r = client.get("/api/conversations")
    assert r.status_code != 401
    assert "Not authenticated" not in r.text
