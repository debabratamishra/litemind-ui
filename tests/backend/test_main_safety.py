"""Startup/shutdown safety and error-response tests for the FastAPI entrypoint.

These guard two data-exposure and data-loss risks in ``backend/main.py``:
uploaded knowledge-base files must survive a restart, and unhandled
exceptions must not reach clients as text.
"""

import logging

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
