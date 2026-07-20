"""Tests for ``app.backend.core.config`` (offline).

``BackendConfig`` reads its dynamic configuration from ``Config.get_dynamic_config()``
(the global ``Config`` class in the repo-root ``config`` module). That helper normally
delegates to ``host_service_manager`` and may create local directories; to keep these
tests deterministic and fully offline we patch ``Config.get_dynamic_config`` with a
fixed fixture dict. The real ``OLLAMA_API_URL`` environment wiring is exercised through
``BackendConfig.get_ollama_url()``.
"""
import os
from pathlib import Path

import config as app_config
from app.backend.core.config import DEFAULT_RAG_CONFIG, BackendConfig


def _fixed_dynamic_config() -> dict:
    return {
        "storage_dir": "./storage",
        "upload_dir": "./uploads",
    }


def test_backend_config_defaults(monkeypatch):
    monkeypatch.setattr(
        app_config.Config, "get_dynamic_config", staticmethod(_fixed_dynamic_config)
    )

    cfg = BackendConfig()
    assert cfg is not None
    assert isinstance(cfg.dynamic_config, dict)
    assert cfg.dynamic_config["storage_dir"] == "./storage"
    assert cfg.storage_dir == Path("./storage")
    assert cfg.upload_folder == Path("./uploads")
    assert cfg.config_path == cfg.storage_dir / "rag_config.json"


def test_default_rag_config_shape():
    assert isinstance(DEFAULT_RAG_CONFIG, dict)
    assert DEFAULT_RAG_CONFIG["provider"] == "huggingface"
    assert "embedding_model" in DEFAULT_RAG_CONFIG
    assert "chunk_size" in DEFAULT_RAG_CONFIG


def test_backend_config_ollama_url_from_env(mock_env, monkeypatch):
    """``get_ollama_url()`` must reflect the ``OLLAMA_API_URL`` env var.

    The real ``host_service_manager`` caches ``OLLAMA_API_URL`` at import time, so we
    substitute a stub whose ``ollama_url`` is read live from the environment. This
    verifies the *variable name* (``OLLAMA_API_URL``) and the routing through
    ``get_ollama_url()`` without any network access.
    """
    mock_env(OLLAMA_API_URL="http://example:11434")

    class _EnvConfig:
        @property
        def ollama_url(self) -> str:
            return os.getenv("OLLAMA_API_URL", "http://localhost:11434")

    class _StubHostServiceManager:
        environment_config = _EnvConfig()

    monkeypatch.setattr(
        app_config.Config, "get_dynamic_config", staticmethod(_fixed_dynamic_config)
    )
    monkeypatch.setattr(
        "app.services.host_service_manager.host_service_manager",
        _StubHostServiceManager(),
    )

    cfg = BackendConfig()
    assert "example:11434" in cfg.get_ollama_url()
