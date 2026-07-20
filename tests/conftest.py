"""Shared pytest fixtures for the LiteMindUI backend test suite.

All tests are offline: external services (Ollama, ChromaDB, LLM providers,
Whisper, Kokoro, SerpAPI, CrewAI, Pipecat) are mocked at their boundary by
the individual test modules. This file only provides cross-cutting helpers.
"""
from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest


@pytest.fixture
def tmp_upload_dir(tmp_path, monkeypatch):
    """A temp upload directory bound to UPLOAD_FOLDER / Config.upload_folder."""
    d = tmp_path / "uploads"
    d.mkdir()
    monkeypatch.setenv("UPLOAD_FOLDER", str(d))
    try:
        import config as app_config

        if hasattr(app_config, "Config"):
            monkeypatch.setattr(app_config.Config, "upload_folder", str(d), raising=False)
    except Exception:
        pass
    return d


@pytest.fixture
def tmp_chroma_dir(tmp_path, monkeypatch):
    """A temp ChromaDB storage directory bound to CHROMA_DB_PATH."""
    d = tmp_path / "chroma"
    d.mkdir()
    monkeypatch.setenv("CHROMA_DB_PATH", str(d))
    return d


@pytest.fixture
def mock_env(monkeypatch):
    """Set/get environment variables for the duration of a test."""

    def _set(**kwargs: str) -> None:
        for k, v in kwargs.items():
            monkeypatch.setenv(k, v)

    return _set


@pytest.fixture
def httpx_mock(monkeypatch):
    """Return a helper that routes all httpx traffic to a handler (offline)."""
    handlers: list[Callable[[httpx.Request], httpx.Response]] = []

    def register(handler: Callable[[httpx.Request], httpx.Response]) -> None:
        handlers.append(handler)

    def _transport(request: httpx.Request) -> httpx.Response:
        for h in handlers:
            resp = h(request)
            if resp is not None:
                return resp
        # Default: fail loudly so an unmocked network call is caught.
        raise AssertionError(f"Unmocked httpx request: {request.method} {request.url}")

    transport = httpx.MockTransport(_transport)
    real_client = httpx.Client
    real_async_client = httpx.AsyncClient

    def _client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    def _async_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", _client)
    monkeypatch.setattr(httpx, "AsyncClient", _async_client)
    return register
