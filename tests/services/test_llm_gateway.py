"""Unit tests for ``app.services.llm_gateway`` (offline).

External boundaries (``litellm`` and the ``ollama`` Python client) are mocked so
no real network/model calls occur. We exercise:

* backend normalization and configuration resolution (params + env),
* the Ollama *native* streaming path (the codebase bypasses LiteLLM for Ollama
  streaming to avoid an upstream ``KeyError('litellm.utils')`` bug),
* the LiteLLM streaming path for ``openrouter`` / ``nvidia_nim``,
* retry / error / config-error handling,
* the small pure helpers (``_extract_stream_text`` etc.).
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services import llm_gateway as gw


# ── normalize_backend ────────────────────────────────────────────────────
def test_normalize_backend_known_canonical():
    assert gw.normalize_backend("Ollama") == "ollama"
    assert gw.normalize_backend("openrouter") == "openrouter"
    assert gw.normalize_backend("NVIDIA_NIM") == "nvidia_nim"


def test_normalize_backend_strips_and_lowercases():
    assert gw.normalize_backend("  OLLAMA ") == "ollama"
    # default when None
    assert gw.normalize_backend(None) == "ollama"


def test_normalize_backend_unknown_raises():
    with pytest.raises(gw.LLMGatewayConfigurationError):
        gw.normalize_backend("bogus")


# ── resolve_backend_config ───────────────────────────────────────────────
def test_resolve_backend_config_ollama_from_params():
    cfg = gw.resolve_backend_config(backend="ollama", model="llama3")
    assert isinstance(cfg, gw.ResolvedLLMConfig)
    assert cfg.backend == "ollama"
    assert cfg.model == "ollama_chat/llama3"
    # api_base comes from host_service_manager (offline-safe) for ollama
    assert cfg.api_base
    assert cfg.api_key is None


def test_resolve_backend_config_ollama_default_model(monkeypatch):
    # Clear any ambient OLLAMA_MODEL / DEFAULT_OLLAMA_MODEL so the hardcoded
    # fallback ("gemma3:1b") is exercised deterministically.
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("DEFAULT_OLLAMA_MODEL", raising=False)
    cfg = gw.resolve_backend_config(backend="ollama", model=None)
    assert cfg.model == "ollama_chat/gemma3:1b"
    assert cfg.backend == "ollama"


def test_resolve_backend_config_strips_known_prefix_idempotently():
    cfg = gw.resolve_backend_config(backend="ollama", model="ollama_chat/llama3")
    assert cfg.model == "ollama_chat/llama3"


def test_resolve_backend_config_openrouter_with_api_key_param():
    cfg = gw.resolve_backend_config(backend="openrouter", model="gpt-4", api_key="k")
    assert cfg.model == "openrouter/gpt-4"
    assert cfg.api_key == "k"
    assert cfg.api_base == gw._OPENROUTER_DEFAULT_API_BASE


def test_resolve_backend_config_openrouter_key_from_env(mock_env):
    mock_env(OPENROUTER_API_KEY="envkey")
    cfg = gw.resolve_backend_config(backend="openrouter", model="gpt-4")
    assert cfg.api_key == "envkey"
    assert cfg.model == "openrouter/gpt-4"


def test_resolve_backend_config_openrouter_missing_key_raises():
    with pytest.raises(gw.LLMGatewayConfigurationError):
        gw.resolve_backend_config(backend="openrouter", model="gpt-4")


def test_resolve_backend_config_nvidia_nim():
    cfg = gw.resolve_backend_config(backend="nvidia_nim", model="llama3", api_key="nk")
    assert cfg.model == "nvidia_nim/llama3"
    assert cfg.api_key == "nk"
    assert cfg.api_base == gw._NVIDIA_NIM_DEFAULT_API_BASE


def test_resolve_backend_config_nvidia_nim_key_from_env(mock_env):
    mock_env(NVIDIA_NIM_API_KEY="nvkey")
    cfg = gw.resolve_backend_config(backend="nvidia_nim", model="llama3")
    assert cfg.api_key == "nvkey"


def test_resolve_backend_config_nvidia_nim_missing_key_raises():
    with pytest.raises(gw.LLMGatewayConfigurationError):
        gw.resolve_backend_config(backend="nvidia_nim", model="llama3")


def test_resolve_backend_config_explicit_api_base_overrides_env(mock_env):
    mock_env(OPENROUTER_API_BASE="https://example.test/v1")
    cfg = gw.resolve_backend_config(backend="openrouter", model="gpt-4", api_key="k", api_base="https://override.test/v1")
    assert cfg.api_base == "https://override.test/v1"


# ── resolve_embedding_config ─────────────────────────────────────────────
def test_resolve_embedding_config_uses_embedding_prefix():
    cfg = gw.resolve_embedding_config(backend="ollama", model="llama3")
    # embedding purpose uses the `ollama/` prefix, completion uses `ollama_chat/`
    assert cfg.model == "ollama/llama3"


def test_resolve_embedding_config_openrouter():
    cfg = gw.resolve_embedding_config(backend="openrouter", model="text-embedding", api_key="k")
    assert cfg.model == "openrouter/text-embedding"


# ── streaming helpers ────────────────────────────────────────────────────
def _make_chunk(content, *, use_message=False):
    chunk = MagicMock()
    choice = MagicMock()
    if use_message:
        msg = MagicMock()
        msg.content = content
        choice.message = msg
        choice.delta = None
    else:
        delta = MagicMock()
        delta.content = content
        choice.delta = delta
        choice.message = None
    chunk.choices = [choice]
    return chunk


def test_extract_stream_text_from_delta():
    assert gw._extract_stream_text(_make_chunk("hello")) == "hello"


def test_extract_stream_text_from_message():
    assert gw._extract_stream_text(_make_chunk("hi", use_message=True)) == "hi"


def test_extract_stream_text_empty_choices():
    chunk = MagicMock()
    chunk.choices = []
    assert gw._extract_stream_text(chunk) == ""


def test_extract_stream_text_list_content():
    chunk = _make_chunk([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
    assert gw._extract_stream_text(chunk) == "ab"


def test_extract_stream_text_none_content():
    assert gw._extract_stream_text(_make_chunk(None)) == ""


def test_coerce_text_handles_scalars_and_lists():
    assert gw._coerce_text(None) == ""
    assert gw._coerce_text(123) == "123"
    assert gw._coerce_text(["x", "y"]) == "xy"


def test_presence_penalty_from_repetition():
    assert gw._presence_penalty_from_repetition(1.0) == 0.0
    assert gw._presence_penalty_from_repetition(0.5) == 0.0
    assert gw._presence_penalty_from_repetition(2.5) == 1.5
    assert gw._presence_penalty_from_repetition(5.0) == 2.0  # clamped to 2.0


def test_strip_known_prefix():
    assert gw._strip_known_prefix("ollama_chat/llama3", ("ollama_chat/", "ollama/")) == "llama3"
    assert gw._strip_known_prefix("plain", ("ollama_chat/", "ollama/")) == "plain"


def test_build_openrouter_headers_from_env(mock_env):
    mock_env(OPENROUTER_SITE_URL="https://x.com", OPENROUTER_APP_NAME="App")
    headers = gw._build_openrouter_headers()
    assert headers["HTTP-Referer"] == "https://x.com"
    assert headers["X-Title"] == "App"


def test_build_openrouter_headers_empty_when_unset():
    assert gw._build_openrouter_headers() == {}


def test_build_error_message_variants():
    ollama_cfg = gw.ResolvedLLMConfig(backend="ollama", model="m")
    assert "Ollama" in gw._build_error_message(ollama_cfg, interrupted=False)
    assert "interrupted" in gw._build_error_message(ollama_cfg, interrupted=True).lower()

    or_cfg = gw.ResolvedLLMConfig(backend="openrouter", model="m")
    assert "OpenRouter" in gw._build_error_message(or_cfg, interrupted=False)

    nim_cfg = gw.ResolvedLLMConfig(backend="nvidia_nim", model="m")
    assert "Nvidia NIM" in gw._build_error_message(nim_cfg, interrupted=False)


# ── get_ollama_api_base fallback ─────────────────────────────────────────
def test_get_ollama_api_base_falls_back_to_config_on_import_error(monkeypatch):
    real_import = __import__

    def _fake_import(name, *args, **kwargs):
        if name == "app.services.host_service_manager":
            raise ImportError("forced")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)
    monkeypatch.setattr(gw.Config, "OLLAMA_API_URL", "http://cfg.host:11434", raising=False)
    assert gw.get_ollama_api_base() == "http://cfg.host:11434"


# ── stream_completion: Ollama native path ───────────────────────────────
async def test_stream_completion_ollama_native_yields_text():
    async def fake_stream(*a, **k):
        yield "hello"
        yield " world"

    with patch.object(gw, "_stream_ollama_native", fake_stream):
        chunks = [
            c async for c in gw.stream_completion(backend="ollama", model="llama3", messages=[])
        ]
    assert "".join(chunks) == "hello world"


async def test_stream_completion_ollama_native_retries_then_errors():
    counter = {"n": 0}

    async def always_fail(*a, **k):
        counter["n"] += 1
        raise RuntimeError("boom")
        yield  # pragma: no cover - makes this an async generator

    with patch.object(gw, "_stream_ollama_native", always_fail), patch.object(
        gw.asyncio, "sleep", AsyncMock()
    ):
        chunks = [
            c async for c in gw.stream_completion(backend="ollama", model="llama3", messages=[])
        ]
    assert len(chunks) == 1
    assert "Ollama" in chunks[0]
    assert counter["n"] == gw._MAX_RETRIES


async def test_stream_completion_ollama_native_retries_on_transient_then_succeeds():
    calls = {"n": 0}

    async def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        yield "recovered"

    with patch.object(gw, "_stream_ollama_native", flaky), patch.object(
        gw.asyncio, "sleep", AsyncMock()
    ):
        chunks = [
            c async for c in gw.stream_completion(backend="ollama", model="llama3", messages=[])
        ]
    assert chunks == ["recovered"]
    assert calls["n"] == 2


# ── stream_completion: LiteLLM path (openrouter / nvidia_nim) ───────────
async def _patched_acompletion(captured: dict, **kwargs):
    # Mirrors the real litellm.acompletion contract: an async function that
    # returns an async-iterable stream of chunks.
    captured.update(kwargs)

    async def _gen():
        yield _make_chunk("Hel")
        yield _make_chunk("lo")

    return _gen()


async def test_stream_completion_openrouter_uses_litellm():
    captured: dict = {}

    async def fake(**kwargs):
        return await _patched_acompletion(captured, **kwargs)

    with patch.object(gw.litellm, "acompletion", side_effect=fake):
        chunks = [
            c
            async for c in gw.stream_completion(
                backend="openrouter", model="gpt-4", api_key="k", messages=[]
            )
        ]
    assert "".join(chunks) == "Hello"
    assert captured["model"] == "openrouter/gpt-4"
    assert captured["api_key"] == "k"
    assert captured["api_base"] == gw._OPENROUTER_DEFAULT_API_BASE
    assert captured["stream"] is True


async def test_stream_completion_openrouter_adds_extra_headers_when_env_set(mock_env):
    mock_env(OPENROUTER_SITE_URL="https://x.com", OPENROUTER_APP_NAME="App")
    captured: dict = {}

    async def fake(**kwargs):
        return await _patched_acompletion(captured, **kwargs)

    with patch.object(gw.litellm, "acompletion", side_effect=fake):
        chunks = [
            c
            async for c in gw.stream_completion(
                backend="openrouter", model="gpt-4", api_key="k", messages=[]
            )
        ]
    assert chunks
    assert captured["extra_headers"]["HTTP-Referer"] == "https://x.com"
    assert captured["extra_headers"]["X-Title"] == "App"


async def test_stream_completion_nvidia_nim_uses_litellm():
    captured: dict = {}

    async def fake(**kwargs):
        return await _patched_acompletion(captured, **kwargs)

    with patch.object(gw.litellm, "acompletion", side_effect=fake):
        chunks = [
            c
            async for c in gw.stream_completion(
                backend="nvidia_nim", model="llama3", api_key="nk", messages=[]
            )
        ]
    assert "".join(chunks) == "Hello"
    assert captured["model"] == "nvidia_nim/llama3"
    assert captured["api_key"] == "nk"
    assert captured["api_base"] == gw._NVIDIA_NIM_DEFAULT_API_BASE


async def test_stream_completion_openrouter_retry_then_error(mock_env):
    mock_env(OPENROUTER_API_KEY="k")
    calls = {"n": 0}

    async def fail(**kwargs):
        calls["n"] += 1
        raise RuntimeError("boom")

    with patch.object(gw.litellm, "acompletion", side_effect=fail), patch.object(
        gw.asyncio, "sleep", AsyncMock()
    ):
        chunks = [
            c async for c in gw.stream_completion(backend="openrouter", model="gpt-4", messages=[])
        ]
    assert calls["n"] == gw._MAX_RETRIES
    assert len(chunks) == 1
    assert "OpenRouter" in chunks[0]


async def test_stream_completion_config_error_yields_message():
    chunks = [
        c async for c in gw.stream_completion(backend="bogus", model="x", messages=[])
    ]
    assert chunks == ["*Unable to process request with current model configuration.*"]


# ── complete_text ────────────────────────────────────────────────────────
async def test_complete_text_collects_stream():
    async def fake_stream(*a, **k):
        yield "a"
        yield "b"

    with patch.object(gw, "_stream_ollama_native", fake_stream):
        out = await gw.complete_text(messages=[], backend="ollama", model="llama3")
    assert out == "ab"


async def test_complete_text_handles_empty_stream():
    async def fake_stream(*a, **k):
        return
        yield  # pragma: no cover

    with patch.object(gw, "_stream_ollama_native", fake_stream):
        out = await gw.complete_text(messages=[], backend="ollama", model="llama3")
    assert out == ""
