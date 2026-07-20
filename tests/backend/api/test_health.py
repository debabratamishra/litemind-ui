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

from app.backend.api import health
from app.backend.core import config as backend_config_module
from app.backend.models.api_models import HealthResponse


async def test_health_check_returns_healthy():
    result = await health.health_check()
    assert isinstance(result, HealthResponse)
    assert result.status == "healthy"


async def test_readiness_check_ready(tmp_path, monkeypatch):
    up = tmp_path / "uploads"
    st = tmp_path / "storage"
    up.mkdir()
    st.mkdir()

    # Stub the RAG service so the import of ``main`` is never exercised.
    fake_main = types.ModuleType("main")
    setattr(fake_main, "rag_service", object())  # non-None => initialised
    monkeypatch.setitem(sys.modules, "main", fake_main)

    monkeypatch.setattr(backend_config_module.backend_config, "upload_folder", up)
    monkeypatch.setattr(backend_config_module.backend_config, "storage_dir", st)
    monkeypatch.setattr(os, "access", lambda p, m: True)

    result = await health.readiness_check()
    assert result["status"] == "ready"
    assert result["checks"]["rag_service"]["status"] == "ready"
    assert result["checks"][up.name]["status"] == "ready"
    assert result["checks"][st.name]["status"] == "ready"


async def test_readiness_check_rag_unavailable(tmp_path, monkeypatch):
    up = tmp_path / "uploads"
    st = tmp_path / "storage"
    up.mkdir()
    st.mkdir()

    fake_main = types.ModuleType("main")
    setattr(fake_main, "rag_service", None)  # not initialised
    monkeypatch.setitem(sys.modules, "main", fake_main)

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

    fake_main = types.ModuleType("main")
    setattr(fake_main, "rag_service", object())
    monkeypatch.setitem(sys.modules, "main", fake_main)

    monkeypatch.setattr(backend_config_module.backend_config, "upload_folder", up)
    monkeypatch.setattr(backend_config_module.backend_config, "storage_dir", st)
    # Directories reported as not accessible => not ready.
    monkeypatch.setattr(os, "access", lambda p, m: False)

    result = await health.readiness_check()
    assert isinstance(result, JSONResponse)
    assert result.status_code == 503
    assert result.body is not None
