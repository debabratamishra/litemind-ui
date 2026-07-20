"""Unit tests for ``app.services.host_service_manager`` (offline).

The external boundary here is ``httpx.AsyncClient`` (used when validating
service connectivity). It is mocked via the shared ``httpx_mock`` fixture so no
real network call ever occurs. Environment selection (Docker vs native) is
driven by the shared ``EnvironmentDetector`` singleton; since that singleton
caches its result, we deterministically control ``is_containerized`` by
monkeypatching the instance attribute before constructing a fresh
``HostServiceManager``.

We also redirect every cache/config directory at a ``tmp_path`` so the tests
never create directories inside the repository or the user's home folder.
"""
from pathlib import Path

import httpx

from app.core.environment import environment as env_detector
from app.services.host_service_manager import (
    EnvironmentConfig,
    HostServiceManager,
    ServiceStatus,
)


def _override_dirs(mgr: HostServiceManager, tmp_path: Path) -> None:
    """Point every config directory at ``tmp_path`` to keep the FS clean."""
    mgr.environment_config = EnvironmentConfig(
        is_containerized=mgr.environment_config.is_containerized,
        ollama_url=mgr.environment_config.ollama_url,
        hf_cache_dir=tmp_path / "hf",
        ollama_cache_dir=tmp_path / "ollama",
        upload_dir=tmp_path / "uploads",
        chroma_db_dir=tmp_path / "chroma",
        storage_dir=tmp_path / "storage",
    )


# ── Dataclasses ────────────────────────────────────────────────────────────
def test_environment_config_fields():
    cfg = EnvironmentConfig(
        is_containerized=False,
        ollama_url="http://localhost:11434",
        hf_cache_dir=Path("/hf"),
        ollama_cache_dir=Path("/ollama"),
        upload_dir=Path("/up"),
        chroma_db_dir=Path("/chroma"),
        storage_dir=Path("/storage"),
    )
    assert cfg.is_containerized is False
    assert str(cfg.ollama_url) == "http://localhost:11434"
    assert cfg.upload_dir == Path("/up")
    assert cfg.chroma_db_dir == Path("/chroma")


def test_service_status_dataclass():
    st = ServiceStatus(
        name="Ollama", url="http://x", is_available=True, response_time_ms=12.5
    )
    assert st.name == "Ollama"
    assert st.is_available is True
    assert st.response_time_ms == 12.5
    assert st.error_message is None


# ── Environment-aware config selection ──────────────────────────────────────
def test_containerized_config_reads_env(monkeypatch):
    monkeypatch.setattr(env_detector, "_is_containerized", True)
    monkeypatch.setenv("OLLAMA_API_URL", "http://ollama:11434")
    monkeypatch.setenv("HF_HOME", "/root/.cache/huggingface")
    monkeypatch.setenv("OLLAMA_MODELS", "/root/.ollama")
    monkeypatch.setenv("UPLOAD_FOLDER", "/app/uploads")
    monkeypatch.setenv("CHROMA_DB_PATH", "/app/chroma_db")
    monkeypatch.setenv("STORAGE_PATH", "/app/storage")

    mgr = HostServiceManager()
    cfg = mgr.environment_config
    assert cfg.is_containerized is True
    assert cfg.ollama_url == "http://ollama:11434"
    assert str(cfg.hf_cache_dir) == "/root/.cache/huggingface"
    assert str(cfg.ollama_cache_dir) == "/root/.ollama"
    assert str(cfg.upload_dir) == "/app/uploads"
    assert str(cfg.chroma_db_dir) == "/app/chroma_db"
    assert str(cfg.storage_dir) == "/app/storage"


def test_native_config_defaults(monkeypatch):
    monkeypatch.setattr(env_detector, "_is_containerized", False)
    monkeypatch.delenv("OLLAMA_API_URL", raising=False)

    mgr = HostServiceManager()
    cfg = mgr.environment_config
    assert cfg.is_containerized is False
    assert cfg.ollama_url == "http://localhost:11434"
    assert cfg.upload_dir == Path("./uploads")
    assert cfg.chroma_db_dir == Path("./chroma_db")


# ── Host cache path resolution ───────────────────────────────────────────────
def test_get_host_cache_paths_keys(monkeypatch):
    monkeypatch.setattr(env_detector, "_is_containerized", False)
    mgr = HostServiceManager()
    paths = mgr.get_host_cache_paths()
    assert set(paths.keys()) == {
        "huggingface_cache",
        "ollama_cache",
        "uploads",
        "chroma_db",
        "storage",
    }
    assert paths["uploads"].endswith("uploads")


# ── Directory creation (FS kept in tmp) ──────────────────────────────────────
def test_ensure_directories_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(env_detector, "_is_containerized", False)
    mgr = HostServiceManager()
    _override_dirs(mgr, tmp_path)

    results = mgr.ensure_directories_exist()
    assert all(results.values())
    for d in (
        mgr.environment_config.hf_cache_dir,
        mgr.environment_config.ollama_cache_dir,
        mgr.environment_config.upload_dir,
        mgr.environment_config.chroma_db_dir,
        mgr.environment_config.storage_dir,
    ):
        assert d.exists()


def test_ensure_host_cache_directories_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(env_detector, "_is_containerized", False)
    mgr = HostServiceManager()

    fake_paths = {
        "huggingface_cache": str(tmp_path / "hf"),
        "ollama_cache": str(tmp_path / "ollama"),
        "uploads": str(tmp_path / "uploads"),
        "chroma_db": str(tmp_path / "chroma"),
        "storage": str(tmp_path / "storage"),
    }
    monkeypatch.setattr(mgr, "get_host_cache_paths", lambda: fake_paths)

    results = mgr.ensure_host_cache_directories_exist()
    assert all(results.values())
    assert (tmp_path / "uploads").exists()
    assert (tmp_path / "chroma").exists()


# ── Dynamic config + system status ───────────────────────────────────────────
def test_get_dynamic_config(tmp_path, monkeypatch):
    monkeypatch.setattr(env_detector, "_is_containerized", True)
    mgr = HostServiceManager()
    _override_dirs(mgr, tmp_path)

    cfg = mgr.get_dynamic_config()
    assert cfg["is_containerized"] is True
    assert cfg["cache_directories_created"]["uploads"] is True


async def test_get_system_status(tmp_path, monkeypatch, httpx_mock):
    monkeypatch.setattr(env_detector, "_is_containerized", False)
    mgr = HostServiceManager()
    _override_dirs(mgr, tmp_path)

    def handler(request):
        return httpx.Response(200, json={"models": []})

    httpx_mock(handler)

    status = await mgr.get_system_status()
    assert status["environment"]["is_containerized"] is False
    assert "Ollama" in status["services"]
    assert status["services"]["Ollama"]["available"] is True
    assert "directories" in status


# ── Connectivity validation (httpx mocked) ───────────────────────────────────
async def test_validate_service_connectivity_available(httpx_mock, monkeypatch):
    mgr = HostServiceManager()

    def handler(request):
        return httpx.Response(200, json={"models": []})

    httpx_mock(handler)

    status = await mgr.validate_service_connectivity("Ollama", "http://localhost:11434")
    assert status.is_available is True
    assert status.name == "Ollama"
    assert status.url == "http://localhost:11434"
    assert status.response_time_ms is not None


async def test_validate_service_connectivity_http_error(httpx_mock, monkeypatch):
    mgr = HostServiceManager()

    def handler(request):
        return httpx.Response(503, text="down")

    httpx_mock(handler)

    status = await mgr.validate_service_connectivity("Ollama", "http://localhost:11434")
    assert status.is_available is False
    assert status.error_message == "HTTP 503"


async def test_validate_service_connectivity_timeout(httpx_mock, monkeypatch):
    mgr = HostServiceManager()

    def handler(request):
        raise httpx.TimeoutException("timed out")

    httpx_mock(handler)

    status = await mgr.validate_service_connectivity("Ollama", "http://localhost:11434")
    assert status.is_available is False
    assert status.error_message == "Connection timeout"


async def test_validate_service_connectivity_connect_error(httpx_mock, monkeypatch):
    mgr = HostServiceManager()

    def handler(request):
        raise httpx.ConnectError("refused")

    httpx_mock(handler)

    status = await mgr.validate_service_connectivity("Ollama", "http://localhost:11434")
    assert status.is_available is False
    assert status.error_message == "Connection refused"


async def test_get_cached_service_status(httpx_mock, monkeypatch):
    mgr = HostServiceManager()

    def handler(request):
        return httpx.Response(200)

    httpx_mock(handler)

    await mgr.validate_service_connectivity("Ollama", "http://localhost:11434")
    cached = mgr.get_cached_service_status("Ollama")
    assert cached is not None
    assert cached.is_available is True
    # Unknown service returns None (not cached).
    assert mgr.get_cached_service_status("Nonexistent") is None


async def test_validate_all_required_services(httpx_mock, monkeypatch):
    mgr = HostServiceManager()

    def handler(request):
        return httpx.Response(200)

    httpx_mock(handler)

    results = await mgr.validate_all_required_services()
    assert "Ollama" in results
    assert isinstance(results["Ollama"], ServiceStatus)
    assert results["Ollama"].is_available is True
