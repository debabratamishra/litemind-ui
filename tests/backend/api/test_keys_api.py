"""API tests for /api/keys CRUD and the provider probe endpoint (offline).

The key vault is replaced with an in-memory fake so no live PostgreSQL is
needed, and httpx traffic is routed through the shared ``httpx_mock``
transport. Keys never echoed: GET /api/keys carries metadata + ``configured``
only — tests assert no value/token field appears anywhere in the response.
"""

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.app.backend.api import keys as keys_api
from backend.app.kernel import router as kernel_router
from backend.main import app

SECRET = "sk-or-never-echo"


class FakeVault:
    """Minimal KeyVault stand-in over an in-memory dict."""

    def __init__(self):
        self._creds: dict[tuple[str, str], dict] = {}

    async def set_credential(self, kind, provider, label, plaintext):
        entry = self._creds.get((kind, provider))
        if entry is None:
            entry = {"id": f"cred-{len(self._creds) + 1}", "kind": kind, "provider": provider}
            self._creds[(kind, provider)] = entry
        entry["label"] = label
        entry["value"] = plaintext
        entry["updated_at"] = "2026-01-01T00:00:00+00:00"
        return entry["id"]

    async def get_credential(self, kind, provider):
        entry = self._creds.get((kind, provider))
        return entry["value"] if entry else None

    async def list_credentials(self):
        return [
            {
                "id": e["id"],
                "kind": e["kind"],
                "label": e["label"],
                "provider": e["provider"],
                "updated_at": e["updated_at"],
            }
            for e in self._creds.values()
        ]

    async def delete_credential(self, credential_id):
        for key, entry in list(self._creds.items()):
            if entry["id"] == credential_id:
                del self._creds[key]
                return True
        return False


@pytest.fixture
def client(monkeypatch):
    vault = FakeVault()
    monkeypatch.setattr(keys_api, "get_key_vault", lambda: vault)
    monkeypatch.setattr(kernel_router, "get_key_vault", lambda: vault)
    for var in (
        "OPENROUTER_API_KEY",
        "NVIDIA_NIM_API_KEY",
        "SERP_API_KEY",
        "OPENROUTER_API_BASE",
        "OPENROUTER_BASE_URL",
        "NVIDIA_NIM_API_BASE",
    ):
        monkeypatch.delenv(var, raising=False)
    return TestClient(app)


def _field_names(obj):
    if isinstance(obj, dict):
        for key, val in obj.items():
            yield key
            yield from _field_names(val)
    elif isinstance(obj, list):
        for item in obj:
            yield from _field_names(item)


def _put_llm_key(client, value=SECRET):
    resp = client.put(
        "/api/keys",
        json={"kind": "llm", "provider": "openrouter", "label": "main", "value": value},
    )
    assert resp.status_code == 200, resp.text
    return resp


# ── PUT / GET / DELETE ────────────────────────────────────────────────────────


def test_put_then_get_lists_key_configured_without_secret(client):
    _put_llm_key(client)

    listed = client.get("/api/keys")
    assert listed.status_code == 200
    rows = listed.json()
    assert len(rows) == 1
    assert rows[0]["kind"] == "llm"
    assert rows[0]["provider"] == "openrouter"
    assert rows[0]["label"] == "main"
    assert rows[0]["configured"] is True


def test_get_keys_never_expose_secret_fields(client):
    _put_llm_key(client)

    response = client.get("/api/keys")
    names = list(_field_names(response.json()))
    assert not {"value", "token", "api_key", "plaintext"} & set(names)
    assert SECRET not in response.text


def test_put_bad_kind_returns_422(client):
    resp = client.put("/api/keys", json={"kind": "chat", "provider": "openrouter", "value": SECRET})
    assert resp.status_code == 422


def test_put_upserts_same_kind_provider(client):
    _put_llm_key(client, "sk-1")
    resp = client.put(
        "/api/keys",
        json={"kind": "llm", "provider": "openrouter", "label": "renamed", "value": "sk-2"},
    )
    assert resp.status_code == 200
    rows = client.get("/api/keys").json()
    assert len(rows) == 1
    assert rows[0]["label"] == "renamed"


def test_delete_removes_key(client):
    credential_id = _put_llm_key(client).json()["id"]

    resp = client.delete(f"/api/keys/{credential_id}")
    assert resp.status_code == 200
    assert client.get("/api/keys").json() == []
    assert client.delete(f"/api/keys/{credential_id}").status_code == 404


def test_delete_malformed_id_returns_404(client):
    assert client.delete("/api/keys/not-a-uuid").status_code == 404


# ── POST /api/kernel/providers/test ──────────────────────────────────────────


def test_probe_ok_returns_latency(client, httpx_mock):
    httpx_mock(lambda req: httpx.Response(200, json={"data": [{"id": "gpt-4o"}]}))
    _put_llm_key(client)

    resp = client.post("/api/kernel/providers/test", json={"capability": "llm"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert isinstance(body["latency_ms"], (int, float))
    assert "error" not in body


def test_probe_sends_bearer_key_to_models_endpoint(client, httpx_mock):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={})

    httpx_mock(handler)
    _put_llm_key(client)

    resp = client.post("/api/kernel/providers/test", json={"capability": "llm"})

    assert resp.json()["ok"] is True
    assert seen["url"] == "https://openrouter.ai/api/v1/models"
    assert seen["auth"] == f"Bearer {SECRET}"


def test_probe_http_error_returns_ok_false_without_secret(client, httpx_mock):
    httpx_mock(lambda req: httpx.Response(401, json={"error": "invalid key"}))
    _put_llm_key(client)

    resp = client.post("/api/kernel/providers/test", json={"capability": "llm"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["error"]
    assert SECRET not in resp.text


def test_probe_unsupported_capability_returns_501(client):
    for capability in ("stt", "tts", "search"):
        resp = client.post("/api/kernel/providers/test", json={"capability": capability})
        assert resp.status_code == 501, capability


def test_probe_unknown_capability_returns_422(client):
    assert client.post("/api/kernel/providers/test", json={"capability": "nope"}).status_code == 422
