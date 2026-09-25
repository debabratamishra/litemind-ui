"""Unit tests for the mounted RAG router handlers.

The RAG router is mounted in ``backend.main.app``. These tests use a throwaway app so the
skill layer and RAG service can be mocked without ChromaDB, model, or network access.
"""

import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.main
from backend.app.backend.api import rag as rag_api
from backend.app.backend.api import security_utils
from backend.app.backend.api.auth_deps import User, get_current_user
from backend.app.services import user_memory_service


def _make_fake_skill(name="standard", chunks=("doc chunk one", "doc chunk two")):
    async def _stream(_request, _rag_service, **_kwargs):
        for c in chunks:
            yield c

    skill = MagicMock()
    skill.name = name
    skill.stream = _stream
    return skill


@pytest.fixture
def rag_client(monkeypatch, tmp_path):
    # The router's endpoints read ``rag_service`` lazily via ``from backend.main import
    # rag_service``; provide a mocked, truthy service.
    rag_service = MagicMock()
    backend.main.rag_service = rag_service

    registry = MagicMock()
    monkeypatch.setattr(rag_api, "rag_skill_registry", registry)

    # `add_document` / `reset_system` are awaited by the route, so make them async.
    rag_service.add_document = AsyncMock()
    rag_service.reset_system = AsyncMock()

    # Duplicate predicates: both default to "not a duplicate".
    rag_service._is_file_already_processed.return_value = (False, "")
    rag_service._is_filename_already_processed.return_value = (False, "")

    # Ingestion writes to backend_config.upload_folder; point it at a temp dir.
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch.setattr(rag_api.backend_config, "upload_folder", upload_dir)

    app = FastAPI()
    app.dependency_overrides[get_current_user] = lambda: User(id="u1", email="u1@example.com")
    monkeypatch.setattr(user_memory_service, "load_memory_block", AsyncMock(return_value=""))
    app.include_router(rag_api.router)

    client = TestClient(app)
    yield client, registry, rag_service


# --------------------------------------------------------------------------- #
# RAG query
# --------------------------------------------------------------------------- #
def test_rag_query_streams_through_skill(rag_client):
    client, registry, _ = rag_client
    registry.resolve.return_value = _make_fake_skill()

    resp = client.post("/api/rag/query", json={"query": "what is X?"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    # Skill chunks are appended with a newline by the route.
    assert "doc chunk one" in resp.text
    assert "doc chunk two" in resp.text


def test_rag_query_passes_rag_service_to_skill(rag_client):
    client, registry, rag_service = rag_client
    captured = {}

    async def _stream(_request, svc, **_kwargs):
        captured["svc"] = svc
        yield "answer"

    skill = MagicMock()
    skill.name = "standard"
    skill.stream = _stream
    registry.resolve.return_value = skill

    resp = client.post("/api/rag/query", json={"query": "q"})
    assert resp.status_code == 200
    # The real ``backend.main.rag_service`` mock must be threaded into the skill.
    assert captured["svc"] is rag_service


def test_rag_query_no_matching_skill_returns_400(rag_client):
    client, registry, _ = rag_client
    registry.resolve.return_value = None

    resp = client.post("/api/rag/query", json={"query": "q"})
    assert resp.status_code == 400
    assert "No compatible RAG skill" in resp.json()["detail"]


def test_rag_query_service_uninitialized_returns_503(rag_client, monkeypatch):
    client, _registry, _ = rag_client
    # Force the lazy ``from backend.main import rag_service`` lookup to be falsy.
    monkeypatch.setattr(backend.main, "rag_service", None)

    resp = client.post("/api/rag/query", json={"query": "q"})
    assert resp.status_code == 503


def test_rag_query_skill_stream_error_yields_generic_message(rag_client):
    client, registry, _ = rag_client

    async def _failing_stream(_request, _rag_service, **_kwargs):
        yield "partial"
        raise RuntimeError("retrieval boom")

    skill = MagicMock()
    skill.name = "standard"
    skill.stream = _failing_stream
    registry.resolve.return_value = skill

    resp = client.post("/api/rag/query", json={"query": "q"})
    assert resp.status_code == 200
    # The route must not leak the raw exception to the client.
    assert "retrieval boom" not in resp.text
    assert "An error occurred while processing your request" in resp.text


# --------------------------------------------------------------------------- #
# Error responses must not carry internal exception text
# --------------------------------------------------------------------------- #
def test_rag_query_unexpected_error_is_generic(rag_client, monkeypatch):
    client, _registry, _ = rag_client
    # The route imports ``load_memory_block`` from the service module at call time.
    monkeypatch.setattr(
        user_memory_service, "load_memory_block", AsyncMock(side_effect=RuntimeError("db password=hunter2"))
    )

    resp = client.post("/api/rag/query", json={"query": "q"})

    assert resp.status_code == 500
    assert "hunter2" not in resp.text


def test_rag_save_config_unexpected_error_is_generic(rag_client, monkeypatch):
    client, _registry, _ = rag_client
    monkeypatch.setattr(rag_api.backend_config, "save_rag_config", MagicMock(side_effect=OSError("disk /srv/full")))

    resp = client.post(
        "/api/rag/save_config",
        json={"provider": "ollama", "embedding_model": "nomic", "chunk_size": 500},
    )

    assert resp.status_code == 500
    assert "/srv/full" not in resp.text


def test_rag_reset_unexpected_error_is_generic(rag_client, monkeypatch):
    client, _registry, rag_service = rag_client
    rag_service.reset_system = AsyncMock(side_effect=RuntimeError("chroma socket closed"))

    resp = client.post("/api/rag/reset")

    assert resp.status_code == 500
    assert "chroma socket closed" not in resp.text


# --------------------------------------------------------------------------- #
# RAG ingestion (upload)
# --------------------------------------------------------------------------- #
def _upload_file(name, content=b"hello world content"):
    return ("files", (name, io.BytesIO(content), "text/plain"))


def test_rag_upload_processes_file(rag_client):
    client, _registry, rag_service = rag_client
    rag_service._is_file_already_processed.return_value = (False, "")
    rag_service.add_document.return_value = {
        "status": "success",
        "message": "Processed doc.txt",
        "chunks_created": 5,
    }

    resp = client.post("/api/rag/upload", files=[_upload_file("doc.txt")])
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["summary"]["successful"] == 1
    assert body["summary"]["total_chunks_created"] == 5
    rag_service.add_document.assert_awaited_once()


def test_rag_upload_duplicate_is_flagged(rag_client):
    client, registry, rag_service = rag_client
    rag_service._is_file_already_processed.return_value = (True, "already indexed")

    resp = client.post("/api/rag/upload", files=[_upload_file("doc.txt")])
    assert resp.status_code == 200
    body = resp.json()
    assert body["summary"]["duplicates"] == 1
    assert body["summary"]["successful"] == 0
    # Duplicate files are not processed.
    rag_service.add_document.assert_not_called()


def test_rag_upload_too_many_files_returns_400(rag_client):
    client, _registry, _rag_service = rag_client

    files = [_upload_file(f"doc{i}.txt") for i in range(51)]
    resp = client.post("/api/rag/upload", files=files)
    assert resp.status_code == 400
    assert "Too many files" in resp.json()["detail"]


def test_rag_upload_rejects_disallowed_extension(rag_client):
    client, registry, rag_service = rag_client

    resp = client.post("/api/rag/upload", files=[_upload_file("evil.exe", b"x")])
    assert resp.status_code == 200
    body = resp.json()
    # The bad file is reported as an error, not processed.
    assert body["summary"]["errors"] == 1
    rag_service.add_document.assert_not_called()


def test_rag_upload_service_uninitialized_returns_503(rag_client, monkeypatch):
    client, registry, _ = rag_client
    monkeypatch.setattr(backend.main, "rag_service", None)

    resp = client.post("/api/rag/upload", files=[_upload_file("doc.txt")])
    assert resp.status_code == 503


# --------------------------------------------------------------------------- #
# Uploading a duplicate must not destroy the stored copy
# --------------------------------------------------------------------------- #
def test_rag_upload_indexed_duplicate_leaves_stored_file_untouched(rag_client):
    client, _registry, rag_service = rag_client
    stored = rag_api.backend_config.upload_folder / "doc.txt"
    stored.write_bytes(b"original stored bytes")
    rag_service._is_filename_already_processed.return_value = (True, "File 'doc.txt' already processed (3 chunks)")

    resp = client.post("/api/rag/upload", files=[_upload_file("doc.txt", b"replacement bytes")])

    assert resp.status_code == 200
    assert resp.json()["summary"]["duplicates"] == 1
    # The previously stored file is still byte-for-byte intact.
    assert stored.read_bytes() == b"original stored bytes"


def test_rag_upload_new_file_is_written(rag_client):
    client, _registry, rag_service = rag_client
    rag_service.add_document.return_value = {"status": "success", "message": "Processed fresh.txt", "chunks_created": 1}

    resp = client.post("/api/rag/upload", files=[_upload_file("fresh.txt", b"fresh bytes")])

    assert resp.status_code == 200
    assert (rag_api.backend_config.upload_folder / "fresh.txt").read_bytes() == b"fresh bytes"


# --------------------------------------------------------------------------- #
# One oversized file must not abort the rest of the batch
# --------------------------------------------------------------------------- #
def test_rag_upload_oversized_file_does_not_abort_batch(rag_client, monkeypatch):
    client, _registry, rag_service = rag_client
    rag_service.add_document.return_value = {
        "status": "success",
        "message": "Processed small.txt",
        "chunks_created": 1,
    }
    # Drive the real size validator with a tiny limit instead of a 100MB payload.
    monkeypatch.setattr(security_utils, "MAX_FILE_SIZE", 4)

    resp = client.post(
        "/api/rag/upload",
        files=[_upload_file("big.txt", b"far more than four bytes"), _upload_file("small.txt", b"ok")],
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["summary"]["errors"] == 1
    assert body["summary"]["successful"] == 1
    by_name = {r["filename"]: r for r in body["results"]}
    assert by_name["big.txt"]["status"] == "error"
    assert "too large" in by_name["big.txt"]["message"].lower()
    assert by_name["small.txt"]["status"] == "success"


# --------------------------------------------------------------------------- #
# Duplicate detection: the index is the single source of truth
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("file_on_disk", [True, False], ids=["indexed-and-on-disk", "indexed-but-removed"])
def test_duplicate_check_agrees_with_upload(rag_client, file_on_disk):
    client, _registry, rag_service = rag_client
    if file_on_disk:
        (rag_api.backend_config.upload_folder / "doc.txt").write_bytes(b"stored bytes")
    rag_service._is_filename_already_processed.return_value = (True, "File 'doc.txt' already processed (3 chunks)")

    preflight = client.post("/api/rag/check-duplicates", json={"filename": "doc.txt"})
    upload = client.post("/api/rag/upload", files=[_upload_file("doc.txt")])

    assert preflight.json()["is_duplicate"] is True
    assert upload.json()["summary"]["duplicates"] == 1


def test_duplicate_check_ignores_untracked_file_on_disk(rag_client):
    """A file on disk that is not in the index is not a duplicate."""
    client, _registry, rag_service = rag_client
    (rag_api.backend_config.upload_folder / "orphan.txt").write_bytes(b"leftover bytes")
    rag_service._is_filename_already_processed.return_value = (False, "")

    preflight = client.post("/api/rag/check-duplicates", json={"filename": "orphan.txt"})

    assert preflight.json()["is_duplicate"] is False


# --------------------------------------------------------------------------- #
# Error responses must not carry internal exception text
# --------------------------------------------------------------------------- #
def test_rag_upload_invalid_filename_message_is_generic(rag_client, monkeypatch):
    client, _registry, _rag_service = rag_client
    monkeypatch.setattr(
        rag_api, "sanitize_filename", MagicMock(side_effect=ValueError("bad name at /Users/secret/uploads/x.txt"))
    )

    resp = client.post("/api/rag/upload", files=[_upload_file("x.txt")])

    assert resp.status_code == 200
    assert resp.json()["summary"]["errors"] == 1
    assert "/Users/secret" not in resp.text


def test_rag_upload_security_failure_message_is_generic(rag_client, monkeypatch):
    client, _registry, _rag_service = rag_client
    upload_mock = MagicMock()
    upload_mock.__truediv__.return_value.resolve.side_effect = OSError("cannot stat /Users/secret/uploads/x.txt")
    monkeypatch.setattr(rag_api.backend_config, "upload_folder", upload_mock)

    resp = client.post("/api/rag/upload", files=[_upload_file("x.txt")])

    assert resp.status_code == 200
    assert resp.json()["summary"]["errors"] == 1
    assert "/Users/secret" not in resp.text


def test_duplicate_check_invalid_filename_message_is_generic(rag_client, monkeypatch):
    client, _registry, _rag_service = rag_client
    monkeypatch.setattr(
        rag_api, "sanitize_filename", MagicMock(side_effect=ValueError("bad name at /Users/secret/uploads/x.txt"))
    )

    resp = client.post("/api/rag/check-duplicates", json={"filename": "x.txt"})

    assert resp.status_code == 200
    assert resp.json()["is_duplicate"] is False
    assert "/Users/secret" not in resp.text


def test_rag_upload_ingest_error_message_is_generic(rag_client):
    client, _registry, rag_service = rag_client
    rag_service.add_document.side_effect = RuntimeError("ingest failed reading /Users/secret/uploads/x.txt")

    resp = client.post("/api/rag/upload", files=[_upload_file("x.txt")])

    assert resp.status_code == 200
    body = resp.json()
    assert body["summary"]["errors"] == 1
    assert "/Users/secret" not in resp.text
    assert body["results"][0]["message"] == "Failed to process file"


# --------------------------------------------------------------------------- #
# Listing files
# --------------------------------------------------------------------------- #
def test_list_rag_files_reports_indexed_state(rag_client):
    client, _registry, rag_service = rag_client
    upload_dir = rag_api.backend_config.upload_folder
    (upload_dir / "indexed.txt").write_bytes(b"hello")
    (upload_dir / "loose.txt").write_bytes(b"world")
    rag_service._is_filename_already_processed.side_effect = lambda name: (
        (True, "indexed") if name == "indexed.txt" else (False, "")
    )

    resp = client.get("/api/rag/files")

    assert resp.status_code == 200
    by_name = {f["filename"]: f for f in resp.json()["files"]}
    assert by_name["indexed.txt"]["indexed"] is True
    assert by_name["loose.txt"]["indexed"] is False


def test_list_rag_files_service_uninitialized_returns_503(rag_client, monkeypatch):
    client, _registry, _rag_service = rag_client
    monkeypatch.setattr(backend.main, "rag_service", None)

    assert client.get("/api/rag/files").status_code == 503
