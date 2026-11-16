from unittest.mock import AsyncMock, patch

import httpx
import pytest

from backend.app.services.voice_pipeline import (
    VoiceConfigurationError,
    get_convai_signed_url,
    resolve_elevenlabs_credentials,
)


class FakeKeyVault:
    def __init__(self, creds=None, rows=None):
        self._creds = creds or {}
        self._rows = rows or []

    async def get_credential(self, kind, provider):
        return self._creds.get((kind, provider))

    async def list_credentials(self):
        return self._rows


async def test_resolve_credentials_explicit_wins(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "env-key")
    monkeypatch.setenv("ELEVENLABS_AGENT_ID", "env-agent")
    k, a = await resolve_elevenlabs_credentials("my-key", "my-agent")
    assert k == "my-key"
    assert a == "my-agent"


async def test_resolve_credentials_vault_wins(monkeypatch):
    import backend.app.kernel.keyvault as kv_mod

    fake_vault = FakeKeyVault(
        creds={
            ("voice", "elevenlabs"): "vault-key",
            ("voice", "elevenlabs_agent_id"): "vault-agent",
        }
    )
    monkeypatch.setattr(kv_mod, "get_key_vault", lambda: fake_vault)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "env-key")
    monkeypatch.setenv("ELEVENLABS_AGENT_ID", "env-agent")

    k, a = await resolve_elevenlabs_credentials()
    assert k == "vault-key"
    assert a == "vault-agent"


async def test_resolve_credentials_env_fallback(monkeypatch):
    import backend.app.kernel.keyvault as kv_mod

    fake_vault = FakeKeyVault()
    monkeypatch.setattr(kv_mod, "get_key_vault", lambda: fake_vault)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "env-key")
    monkeypatch.setenv("ELEVENLABS_AGENT_ID", "env-agent")

    k, a = await resolve_elevenlabs_credentials()
    assert k == "env-key"
    assert a == "env-agent"


async def test_resolve_credentials_missing_key_raises(monkeypatch):
    import backend.app.kernel.keyvault as kv_mod

    fake_vault = FakeKeyVault()
    monkeypatch.setattr(kv_mod, "get_key_vault", lambda: fake_vault)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setenv("ELEVENLABS_AGENT_ID", "agent-123")

    with pytest.raises(VoiceConfigurationError, match="ElevenLabs API key is required"):
        await resolve_elevenlabs_credentials()


async def test_resolve_credentials_missing_agent_raises(monkeypatch):
    import backend.app.kernel.keyvault as kv_mod

    fake_vault = FakeKeyVault()
    monkeypatch.setattr(kv_mod, "get_key_vault", lambda: fake_vault)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "xi-key")
    monkeypatch.delenv("ELEVENLABS_AGENT_ID", raising=False)

    with pytest.raises(VoiceConfigurationError, match="ElevenLabs Agent ID is required"):
        await resolve_elevenlabs_credentials()


async def test_get_convai_signed_url_success():
    mock_resp = httpx.Response(
        200,
        json={"signed_url": "wss://api.elevenlabs.io/v1/convai/conversation?token=xyz"},
        request=httpx.Request("GET", "https://api.elevenlabs.io"),
    )
    with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_resp)):
        url = await get_convai_signed_url("agent-123", "xi-key")
        assert url == "wss://api.elevenlabs.io/v1/convai/conversation?token=xyz"


async def test_get_convai_signed_url_401_invalid_key():
    mock_resp = httpx.Response(
        401,
        text="Unauthorized",
        request=httpx.Request("GET", "https://api.elevenlabs.io"),
    )
    with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_resp)):
        with pytest.raises(VoiceConfigurationError, match="Invalid ElevenLabs API key"):
            await get_convai_signed_url("agent-123", "bad-key")


async def test_get_convai_signed_url_404_agent_not_found():
    mock_resp = httpx.Response(
        404,
        text="Not found",
        request=httpx.Request("GET", "https://api.elevenlabs.io"),
    )
    with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_resp)):
        with pytest.raises(VoiceConfigurationError, match="ElevenLabs Agent 'agent-123' not found"):
            await get_convai_signed_url("agent-123", "xi-key")
