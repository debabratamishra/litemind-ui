"""Unit tests for the kernel provider router (offline).

The key vault is replaced with an in-memory fake so no live PostgreSQL is
needed. Resolution order under test: explicit request param > key vault > env
var, and ``GET /api/kernel/providers`` reports ``configured: true`` only when
a vault credential or env key/endpoint is actually present.
"""

import pytest
from fastapi.testclient import TestClient

from backend.app.kernel import router as kernel_router
from backend.app.kernel.router import CAPABILITIES
from backend.main import app


class FakeVault:
    """Minimal KeyVault stand-in: get_credential over a dict."""

    def __init__(self, creds=None):
        self._creds = creds or {}

    async def get_credential(self, kind, provider):
        return self._creds.get((kind, provider))


@pytest.fixture
def fake_vault(monkeypatch):
    def _install(creds=None):
        fake = FakeVault(creds)
        monkeypatch.setattr(kernel_router, "get_key_vault", lambda: fake)
        return fake

    return _install


@pytest.fixture
def clean_env(monkeypatch):
    for var in (
        "OPENROUTER_API_KEY",
        "NVIDIA_NIM_API_KEY",
        "OPENROUTER_API_BASE",
        "OPENROUTER_BASE_URL",
        "NVIDIA_NIM_API_BASE",
        "SERP_API_KEY",
        "ELEVENLABS_API_KEY",
        "DEFAULT_OLLAMA_MODEL",
        "OLLAMA_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)


# ── resolve_provider_config: param > vault > env ────────────────────────────


async def test_param_api_key_wins_over_vault_and_env(fake_vault, clean_env, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    fake_vault({("llm", "openrouter"): "vault-key"})
    cfg = await kernel_router.resolve_provider_config("llm", api_key="k")
    assert cfg.api_key == "k"


async def test_vault_key_beats_env(fake_vault, clean_env, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    fake_vault({("llm", "openrouter"): "vault-key"})
    cfg = await kernel_router.resolve_provider_config("llm")
    assert cfg.backend == "openrouter"
    assert cfg.api_key == "vault-key"


async def test_env_key_selects_provider_when_vault_empty(fake_vault, clean_env, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    fake_vault({})
    cfg = await kernel_router.resolve_provider_config("llm")
    assert cfg.backend == "openrouter"
    assert cfg.api_key == "env-key"


async def test_ollama_default_without_key(fake_vault, clean_env):
    fake_vault({})
    cfg = await kernel_router.resolve_provider_config("llm")
    assert cfg.backend == "ollama"
    assert cfg.api_key is None
    assert cfg.api_base
    assert cfg.model


async def test_search_env_key_returns_key_without_param_or_vault(fake_vault, clean_env, monkeypatch):
    monkeypatch.setenv("SERP_API_KEY", "serp-env-key")
    fake_vault({})
    cfg = await kernel_router.resolve_provider_config("search")
    assert cfg.backend == "serpapi"
    assert cfg.api_key == "serp-env-key"


async def test_elevenlabs_vault_key_configures_stt_and_tts(fake_vault, clean_env):
    fake_vault({("tts", "elevenlabs"): "xi-vault-key"})
    stt_cfg = await kernel_router.resolve_provider_config("stt")
    tts_cfg = await kernel_router.resolve_provider_config("tts")
    assert stt_cfg.backend == "elevenlabs"
    assert stt_cfg.api_key == "xi-vault-key"
    assert tts_cfg.backend == "elevenlabs"
    assert tts_cfg.api_key == "xi-vault-key"


async def test_elevenlabs_env_key_selects_stt_and_tts(fake_vault, clean_env, monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "xi-env-key")
    fake_vault({})
    stt_cfg = await kernel_router.resolve_provider_config("stt")
    assert stt_cfg.backend == "elevenlabs"
    assert stt_cfg.api_key == "xi-env-key"


# ── GET /api/kernel/providers ────────────────────────────────────────────────


def test_providers_endpoint_configured_flags(fake_vault, clean_env, monkeypatch):
    fake_vault({("embeddings", "openrouter"): "emb-key"})
    monkeypatch.setenv("SERP_API_KEY", "serp-key")

    body = TestClient(app).get("/api/kernel/providers").json()

    assert set(body) == set(CAPABILITIES)
    assert body["llm"]["provider"] == "ollama"
    assert body["llm"]["configured"] is False
    assert body["embeddings"]["provider"] == "openrouter"
    assert body["embeddings"]["configured"] is True
    assert body["search"]["provider"] == "serpapi"
    assert body["search"]["configured"] is True
    assert body["vlm"]["configured"] is False
    assert body["stt"]["configured"] is False
    assert body["tts"]["configured"] is False
    for cap in CAPABILITIES:
        assert "model" in body[cap]
