"""Unit tests for ``app/backend/api/rag.py`` route handlers.

The RAG router is **not** mounted in ``main.app`` (the RAG endpoints served by
the running app are duplicated inline in ``main.py``). To exercise the real
``app/backend/api/rag.py`` handlers we mount that router on a throwaway
``FastAPI`` app. The skill layer (``rag_skill_registry``) and the RAG service
(``main.rag_service``) are mocked at their boundaries so no ChromaDB / model /
network access occurs.
"""

import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import main
from app.backend.api import rag as rag_api


def _make_fake_skill(name="standard", chunks=("doc chunk one", "doc chunk two")):
    async def _stream(request, rag_service):
        for c in chunks:
            yield c

    skill = MagicMock()
    skill.name = name
    skill.stream = _stream
    return skill


@pytest.fixture
def rag_client(monkeypatch, tmp_path):
    # The router's endpoints read ``rag_service`` lazily via ``from main import
    # rag_service``; provide a mocked, truthy service.
    rag_service = MagicMock()
    main.rag_service = rag_service

    registry = MagicMock()
    monkeypatch.setattr(rag_api, "rag_skill_registry", registry)

    # `add_document` / `reset_system` are awaited by the route, so make them async.
    rag_service.add_document = AsyncMock()
    rag_service.reset_system = AsyncMock()

    # Ingestion writes to backend_config.upload_folder; point it at a temp dir.
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch.setattr(rag_api.backend_config, "upload_folder", upload_dir)

    app = FastAPI()
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

    async def _stream(request, svc):
        captured["svc"] = svc
        yield "answer"

    skill = MagicMock()
    skill.name = "standard"
    skill.stream = _stream
    registry.resolve.return_value = skill

    resp = client.post("/api/rag/query", json={"query": "q"})
    assert resp.status_code == 200
    # The real ``main.rag_service`` mock must be threaded into the skill.
    assert captured["svc"] is rag_service


def test_rag_query_no_matching_skill_returns_400(rag_client):
    client, registry, _ = rag_client
    registry.resolve.return_value = None

    resp = client.post("/api/rag/query", json={"query": "q"})
    assert resp.status_code == 400
    assert "No compatible RAG skill" in resp.json()["detail"]


def test_rag_query_service_uninitialized_returns_503(rag_client, monkeypatch):
    client, registry, _ = rag_client
    # Force the lazy ``from main import rag_service`` lookup to be falsy.
    monkeypatch.setattr(main, "rag_service", None)

    resp = client.post("/api/rag/query", json={"query": "q"})
    assert resp.status_code == 503


def test_rag_query_skill_stream_error_yields_generic_message(rag_client):
    client, registry, _ = rag_client

    async def _failing_stream(request, rag_service):
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
# RAG ingestion (upload)
# --------------------------------------------------------------------------- #
def _upload_file(name, content=b"hello world content"):
    return ("files", (name, io.BytesIO(content), "text/plain"))


def test_rag_upload_processes_file(rag_client):
    client, registry, rag_service = rag_client
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
    client, registry, rag_service = rag_client

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
    monkeypatch.setattr(main, "rag_service", None)

    resp = client.post("/api/rag/upload", files=[_upload_file("doc.txt")])
    assert resp.status_code == 503
