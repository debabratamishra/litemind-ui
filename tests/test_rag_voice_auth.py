"""
Auth-gate tests for the RAG query and voice offer endpoints.

These endpoints live on the main FastAPI app (main.py). We verify they return
401 when no user is authenticated, proving the protection is in place.
"""

import contextlib

from fastapi.testclient import TestClient

from app.backend.api.auth_deps import User, get_current_user
from main import app


@contextlib.contextmanager
def _client(authed: bool):
    # Save/restore the global override so this module doesn't clobber the
    # default set by other test modules (e.g. test_chat / test_voice).
    saved = app.dependency_overrides.get(get_current_user)
    if authed:
        app.dependency_overrides[get_current_user] = lambda: User(id="u1", email="u1@x.com")
    else:
        app.dependency_overrides.pop(get_current_user, None)
    try:
        yield TestClient(app)
    finally:
        if saved is None:
            app.dependency_overrides.pop(get_current_user, None)
        else:
            app.dependency_overrides[get_current_user] = saved


def test_rag_query_requires_auth():
    with _client(authed=False) as client:
        r = client.post("/api/rag/query", json={"query": "hello"})
    assert r.status_code == 401


def test_voice_offer_requires_auth():
    with _client(authed=False) as client:
        r = client.post("/api/voice/offer", json={"pc_id": "x", "sdp": "y", "type": "offer"})
    assert r.status_code == 401
