"""Unit tests for ``app.backend.api.health`` (offline).

``health_check`` returns a simple healthy payload. ``readiness_check`` inspects
the RAG service and critical directories; we stub both boundaries (the ``main``
module's ``rag_service`` and ``backend_config`` directories + ``os.access``)
so no real service/network is touched.
"""
import json
import os
import sys
import types

from fastapi.responses import JSONResponse

from backend.app.backend.api import health
from backend.app.backend.core import config as backend_config_module
from backend.app.backend.models.api_models import HealthResponse


async def test_health_check_returns_healthy():
    result = await health.health_check()
    assert isinstance(result, HealthResponse)
    assert result.status == "healthy"
    # Clients key on the service name; it is part of the public contract.
    assert result.service == "LiteMindUI API"


async def test_readiness_check_ready(tmp_path, monkeypatch):
    up = tmp_path / "uploads"
    st = tmp_path / "storage"
    up.mkdir()
    st.mkdir()

    # Stub the RAG service so the import of ``main`` is never exercised.
    fake_main = types.ModuleType("backend.main")
    setattr(fake_main, "rag_service", object())  # non-None => initialised
    monkeypatch.setitem(sys.modules, "backend.main", fake_main)

    monkeypatch.setattr(backend_config_module.backend_config, "upload_folder", up)
    monkeypatch.setattr(backend_config_module.backend_config, "storage_dir", st)
    monkeypatch.setattr(os, "access", lambda p, m: True)

    result = await health.readiness_check()
    assert result["status"] == "ready"
    assert result["checks"]["rag_service"]["status"] == "ready"
    assert result["checks"][up.name]["status"] == "ready"


async def test_readiness_check_rag_unavailable(tmp_path, monkeypatch):
    up = tmp_path / "uploads"
    st = tmp_path / "storage"
    up.mkdir()
    st.mkdir()

    fake_main = types.ModuleType("backend.main")
    setattr(fake_main, "rag_service", None)  # not initialised
    monkeypatch.setitem(sys.modules, "backend.main", fake_main)

    monkeypatch.setattr(backend_config_module.backend_config, "upload_folder", up)
    monkeypatch.setattr(backend_config_module.backend_config, "storage_dir", st)
    monkeypatch.setattr(os, "access", lambda p, m: True)

    result = await health.readiness_check()
    assert isinstance(result, JSONResponse)
    assert result.status_code == 503
    body = result.body
    # JSONResponse body is bytes; decode to inspect.
    payload = json.loads(body)
    assert payload["status"] == "not_ready"
    assert payload["checks"]["rag_service"]["status"] == "failed"


async def test_readiness_check_dir_not_writable(tmp_path, monkeypatch):
    up = tmp_path / "uploads"
    st = tmp_path / "storage"
    up.mkdir()
    st.mkdir()

    fake_main = types.ModuleType("backend.main")
    setattr(fake_main, "rag_service", object())
    monkeypatch.setitem(sys.modules, "backend.main", fake_main)

    monkeypatch.setattr(backend_config_module.backend_config, "upload_folder", up)
    monkeypatch.setattr(backend_config_module.backend_config, "storage_dir", st)
    # Directories reported as not accessible => not ready.
    monkeypatch.setattr(os, "access", lambda p, m: False)

    result = await health.readiness_check()
    assert isinstance(result, JSONResponse)
    assert result.status_code == 503
    assert result.body is not None


async def test_readiness_check_missing_storage_dir_is_still_ready(tmp_path, monkeypatch):
    """A degraded storage path is not the same as an unready process.

    The probe answers "can this process serve?", which depends on the upload
    directory (writable, and the RAG write path) — not on the storage dir.
    """
    up = tmp_path / "uploads"
    up.mkdir()
    missing_storage = tmp_path / "storage-does-not-exist"

    fake_main = types.ModuleType("backend.main")
    setattr(fake_main, "rag_service", object())
    monkeypatch.setitem(sys.modules, "backend.main", fake_main)

    monkeypatch.setattr(backend_config_module.backend_config, "upload_folder", up)
    monkeypatch.setattr(backend_config_module.backend_config, "storage_dir", missing_storage)
    monkeypatch.setattr(os, "access", lambda p, m: True)

    result = await health.readiness_check()

    assert not isinstance(result, JSONResponse)
    assert result["status"] == "ready"
