"""Startup/shutdown safety and error-response tests for the FastAPI entrypoint.

These guard two data-exposure and data-loss risks in ``backend/main.py``:
uploaded knowledge-base files must survive a restart, and unhandled
exceptions must not reach clients as text.
"""

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.main as main_module


# --------------------------------------------------------------------------- #
# Upload folder preservation
# --------------------------------------------------------------------------- #
def test_prepare_upload_folder_preserves_existing_files(tmp_path):
    from backend.main import _prepare_upload_folder

    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    marker = upload_dir / "keep-me.txt"
    marker.write_text("keep me")

    _prepare_upload_folder(upload_dir, is_containerized=False)

    assert marker.exists()
    assert marker.read_text() == "keep me"


def test_prepare_upload_folder_creates_missing_directory(tmp_path):
    from backend.main import _prepare_upload_folder

    upload_dir = tmp_path / "nested" / "uploads"

    _prepare_upload_folder(upload_dir, is_containerized=False)

    assert upload_dir.is_dir()


def test_prepare_upload_folder_applies_container_permissions(tmp_path):
    from backend.main import _prepare_upload_folder

    upload_dir = tmp_path / "uploads"

    _prepare_upload_folder(upload_dir, is_containerized=True)

    assert upload_dir.is_dir()
    assert upload_dir.stat().st_mode & 0o777 == 0o755


# --------------------------------------------------------------------------- #
# Shutdown must not destroy the knowledge base
# --------------------------------------------------------------------------- #
async def test_shutdown_does_not_reset_rag_system(monkeypatch):
    """Shutdown must never clear the index: it is user data on a persistent volume.

    ``RAGService.reset_system()`` empties the Chroma collection and the
    ``processed_files`` / ``file_hashes`` / chunk maps, so calling it while
    shutting down silently destroys every indexed document on each restart.
    The guard is the module-global lookup below: re-adding ``reset_system`` to
    the shutdown path (which reads ``rag_service`` the same way) fails this test.
    """
    rag_service = MagicMock()
    rag_service.reset_system = AsyncMock()
    monkeypatch.setattr(main_module, "rag_service", rag_service)

    await main_module._shutdown_cleanup()

    rag_service.reset_system.assert_not_awaited()


async def test_lifespan_shutdown_does_not_reset_rag_system(monkeypatch, tmp_path):
    """Drive the whole lifespan, so every shutdown path is covered.

    The helper test above pins ``_shutdown_cleanup`` only: re-adding
    ``reset_system()`` *inline in lifespan* would slip past it. This test runs
    startup and shutdown end to end with the startup side-effects faked out, so
    the call cannot come back anywhere in the shutdown path without failing here.
    """
    rag_service = MagicMock()
    rag_service.reset_system = AsyncMock()
    memory_store = MagicMock()
    memory_store.init_schema = AsyncMock()
    config_info = {
        "is_containerized": False,
        "storage_dir": str(tmp_path),
        "ollama_url": "http://localhost:11434",
    }

    # Patching the module global records its old value, so teardown restores
    # whatever lifespan rebinds it to.
    monkeypatch.setattr(main_module, "rag_service", rag_service)
    monkeypatch.setattr(main_module, "RAGService", lambda: rag_service)
    monkeypatch.setattr(main_module, "UPLOAD_FOLDER", tmp_path / "uploads")
    monkeypatch.setattr(main_module, "load_rag_config", lambda: dict(main_module.DEFAULT_RAG_CONFIG))
    monkeypatch.setattr(main_module, "create_embedding_function", MagicMock())
    monkeypatch.setattr(main_module, "get_user_memory_store", lambda: memory_store)
    monkeypatch.setattr(main_module.Config, "get_dynamic_config", classmethod(lambda cls: dict(config_info)))
    monkeypatch.setenv("PRELOAD_SPEECH_MODELS", "0")

    app = FastAPI()
    async with main_module.lifespan(app):
        assert app.state.start_time > 0

    rag_service.reset_system.assert_not_awaited()


# --------------------------------------------------------------------------- #
# Root compatibility shim
# --------------------------------------------------------------------------- #
def test_root_main_shim_dispatches_to_run(monkeypatch):
    """``python main.py`` must still start the server.

    The root ``main.py`` is a compatibility shim. Without the ``run`` re-export
    and its ``__main__`` guard it exits 0 having started nothing, which is a
    silent failure for anyone using the documented entrypoint.
    """
    calls: list[str] = []
    monkeypatch.setattr(main_module, "run", lambda: calls.append("run"))

    shim_path = Path(__file__).resolve().parents[2] / "main.py"
    # Executing with ``__name__ == "__main__"`` is what ``python main.py`` does.
    exec(compile(shim_path.read_text(), str(shim_path), "exec"), {"__name__": "__main__", "__file__": str(shim_path)})

    assert calls == ["run"]


def test_root_main_shim_reexports_app():
    shim_path = Path(__file__).resolve().parents[2] / "main.py"
    namespace: dict = {"__name__": "root_main_shim", "__file__": str(shim_path)}
    exec(compile(shim_path.read_text(), str(shim_path), "exec"), namespace)

    assert namespace["app"] is main_module.app
    assert namespace["run"] is main_module.run


# --------------------------------------------------------------------------- #
# Generic internal-error responses
# --------------------------------------------------------------------------- #
@pytest.fixture
def boom_client():
    """A minimal app carrying the production 500 handler plus a failing route."""
    app = FastAPI()
    app.add_exception_handler(500, main_module.internal_error_handler)

    @app.get("/__test__/boom")
    async def boom():
        raise RuntimeError("secret detail")

    # ``raise_server_exceptions=False`` is what a real deployment does: the
    # server answers the client instead of propagating the traceback.
    return TestClient(app, raise_server_exceptions=False)


def test_internal_errors_are_generic(boom_client):
    response = boom_client.get("/__test__/boom")

    assert response.status_code == 500
    assert "secret detail" not in response.text


def test_internal_errors_are_logged_server_side(boom_client, caplog):
    with caplog.at_level(logging.ERROR):
        boom_client.get("/__test__/boom")

    assert "secret detail" in caplog.text
