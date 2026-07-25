"""Unit tests for ``app/backend/api/chat.py`` route handlers.

The chat route delegates to the LLM gateway (``complete_text`` /
``stream_completion``) and to the pluggable chat skill layer
(``chat_skill_registry``). All of those boundaries are mocked here so no real
model, network, or SerpAPI call happens offline.

Chat endpoints are reachable through ``main.app`` (the chat router is mounted
there). We patch the names bound inside the ``chat`` module because FastAPI
imports them directly at module load.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.backend.api import chat as chat_api
from app.backend.api.auth_deps import User, get_current_user
from app.skills.base import SkillValidationResult
from main import app

# Chat endpoints now require authentication; satisfy it for these unit tests.
app.dependency_overrides[get_current_user] = lambda: User(id="u1", email="u1@x.com")

client = TestClient(app)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
async def _fake_stream(*_args, **_kwargs):
    """Stand-in async generator for ``stream_completion``."""
    for chunk in ["Hello", " world"]:
        yield chunk


def _make_fake_skill(name="web_search", ok=True):
    """Build a fake chat skill whose ``stream`` is an async generator."""

    async def _stream(request):
        yield "search result part 1"
        yield "search result part 2"

    skill = MagicMock()
    skill.name = name
    skill.validate.return_value = SkillValidationResult(ok=ok, message="n/a" if ok else "bad token")
    skill.stream = _stream
    return skill


# --------------------------------------------------------------------------- #
# Non-streaming chat
# --------------------------------------------------------------------------- #
def test_chat_endpoint_returns_response(monkeypatch):
    monkeypatch.setattr(chat_api, "complete_text", AsyncMock(return_value="Hello there"))

    resp = client.post("/api/chat", json={"message": "hi", "model": "gemma3:1b"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["response"] == "Hello there"
    assert body["model"] == "gemma3:1b"


def test_chat_endpoint_default_model(monkeypatch):
    monkeypatch.setattr(chat_api, "complete_text", AsyncMock(return_value="ok"))

    resp = client.post("/api/chat", json={"message": "hi"})
    assert resp.status_code == 200
    # Default model falls back to the string "default" when not supplied.
    assert resp.json()["model"] == "default"


def test_chat_endpoint_validation_error_when_message_missing():
    # ``message`` is required by ChatRequestEnhanced -> 422 from FastAPI.
    resp = client.post("/api/chat", json={"model": "gemma3:1b"})
    assert resp.status_code == 422


def test_chat_endpoint_internal_error_returns_500(monkeypatch):
    monkeypatch.setattr(chat_api, "complete_text", AsyncMock(side_effect=RuntimeError("boom")))

    resp = client.post("/api/chat", json={"message": "hi"})
    assert resp.status_code == 500


def test_chat_endpoint_voice_mode_uses_short_max_tokens(monkeypatch):
    captured = {}

    async def _capture(*args, **kwargs):
        captured["max_tokens"] = kwargs.get("max_tokens")
        return "voice reply"

    monkeypatch.setattr(chat_api, "complete_text", _capture)

    resp = client.post("/api/chat", json={"message": "hi", "is_voice_mode": True})
    assert resp.status_code == 200
    assert captured["max_tokens"] == 300


def test_chat_endpoint_generative_ui_raises_min_tokens(monkeypatch):
    captured = {}

    async def _capture(*args, **kwargs):
        captured["max_tokens"] = kwargs.get("max_tokens")
        return "ui reply"

    monkeypatch.setattr(chat_api, "complete_text", _capture)

    resp = client.post(
        "/api/chat",
        json={"message": "hi", "enable_generative_ui": True, "max_tokens": 100},
    )
    assert resp.status_code == 200
    # GenUI mode forces at least GENUI_MIN_MAX_TOKENS (16384).
    assert captured["max_tokens"] == 16_384


# --------------------------------------------------------------------------- #
# Streaming chat
# --------------------------------------------------------------------------- #
def test_chat_stream_returns_sse_chunks(monkeypatch):
    monkeypatch.setattr(chat_api, "stream_completion", _fake_stream)

    resp = client.post("/api/chat/stream", json={"message": "hi"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    lines = [ln for ln in resp.text.split("\n\n") if ln.strip()]
    assert len(lines) == 2
    # Each SSE frame is `data: {json}` with the chunk payload.
    assert json.loads(lines[0][len("data: ") :])["chunk"] == "Hello"
    assert json.loads(lines[1][len("data: ") :])["chunk"] == " world"


def test_chat_stream_forwards_generative_ui_fenced_block(monkeypatch):
    """The stream must transparently forward Generative-UI ``ui:`` fences.

    The chat endpoint does not parse the fences; it forwards whatever the LLM
    emits. We verify a ``ui:data_table`` fenced block survives the SSE framing.
    """

    async def _ui_stream(*_args, **_kwargs):
        yield "Here is a table:\n"
        yield "```ui:data_table\n"
        yield '{"columns": ["A", "B"]}\n'
        yield "```\n"

    monkeypatch.setattr(chat_api, "stream_completion", _ui_stream)

    resp = client.post("/api/chat/stream", json={"message": "hi", "enable_generative_ui": True})
    assert resp.status_code == 200
    assert "ui:data_table" in resp.text
    assert "```ui:data_table" in resp.text


def test_chat_stream_validation_error_when_message_missing():
    resp = client.post("/api/chat/stream", json={"model": "gemma3:1b"})
    assert resp.status_code == 422


def test_chat_stream_internal_error_yields_error_frame(monkeypatch):
    async def _bad_stream(*_args, **_kwargs):
        yield "partial"
        raise RuntimeError("boom")

    monkeypatch.setattr(chat_api, "stream_completion", _bad_stream)

    resp = client.post("/api/chat/stream", json={"message": "hi"})
    assert resp.status_code == 200
    # The generator catches the error and emits an error SSE frame.
    assert "An internal error occurred" in resp.text


# --------------------------------------------------------------------------- #
# Web-search chat (skill routing + fallbacks)
# --------------------------------------------------------------------------- #
def test_chat_web_search_routes_to_skill(monkeypatch):
    registry = MagicMock()
    registry.resolve.return_value = _make_fake_skill(ok=True)
    monkeypatch.setattr(chat_api, "chat_skill_registry", registry)

    resp = client.post("/api/chat/web-search", json={"message": "latest news", "use_web_search": True})
    assert resp.status_code == 200
    # Skill chunks are appended with a trailing newline.
    assert "search result part 1" in resp.text
    assert "search result part 2" in resp.text
    registry.resolve.assert_called_once()


def test_chat_web_search_routes_to_standard_when_not_requested(monkeypatch):
    monkeypatch.setattr(chat_api, "stream_completion", _fake_stream)

    resp = client.post("/api/chat/web-search", json={"message": "hi", "use_web_search": False})
    assert resp.status_code == 200
    assert "Hello" in resp.text


def test_chat_web_search_no_skill_falls_back_to_local(monkeypatch):
    registry = MagicMock()
    registry.resolve.return_value = None
    monkeypatch.setattr(chat_api, "chat_skill_registry", registry)
    monkeypatch.setattr(chat_api, "stream_completion", _fake_stream)

    resp = client.post("/api/chat/web-search", json={"message": "hi", "use_web_search": True})
    assert resp.status_code == 200
    # No skill matched -> standard local streaming fallback.
    assert "Hello" in resp.text


def test_chat_web_search_invalid_skill_falls_back_to_local(monkeypatch):
    registry = MagicMock()
    # Skill matches but validation fails (e.g. missing SerpAPI token).
    registry.resolve.return_value = _make_fake_skill(ok=False)
    monkeypatch.setattr(chat_api, "chat_skill_registry", registry)
    monkeypatch.setattr(chat_api, "stream_completion", _fake_stream)

    resp = client.post("/api/chat/web-search", json={"message": "hi", "use_web_search": True})
    assert resp.status_code == 200
    # Error notice is streamed, then the local fallback kicks in.
    assert "SerpAPI token is required" in resp.text
    assert "Hello" in resp.text


def test_chat_web_search_skill_failure_falls_back_to_local(monkeypatch):
    registry = MagicMock()
    skill = _make_fake_skill(ok=True)

    async def _failing_stream(request):
        yield "partial"
        raise RuntimeError("skill boom")

    skill.stream = _failing_stream
    registry.resolve.return_value = skill
    monkeypatch.setattr(chat_api, "chat_skill_registry", registry)
    monkeypatch.setattr(chat_api, "stream_completion", _fake_stream)

    resp = client.post("/api/chat/web-search", json={"message": "hi", "use_web_search": True})
    assert resp.status_code == 200
    assert "Error during web search" in resp.text
    # Local fallback still produced output.
    assert "Hello" in resp.text


# --------------------------------------------------------------------------- #
# SerpAPI token status
# --------------------------------------------------------------------------- #
def test_serp_status_valid(monkeypatch):
    svc = MagicMock()
    svc.validate_token.return_value = {"valid": True, "message": "Token OK"}
    monkeypatch.setattr(chat_api, "WebSearchService", MagicMock(return_value=svc))

    resp = client.post("/api/chat/serp-status", json={})
    assert resp.status_code == 200
    assert resp.json()["status"] == "valid"


def test_serp_status_invalid(monkeypatch):
    svc = MagicMock()
    svc.validate_token.return_value = {"valid": False, "message": "bad key"}
    monkeypatch.setattr(chat_api, "WebSearchService", MagicMock(return_value=svc))

    resp = client.post("/api/chat/serp-status", json={})
    assert resp.status_code == 200
    assert resp.json()["status"] == "invalid"


def test_serp_status_with_provided_key(monkeypatch):
    svc = MagicMock()
    svc.validate_token.return_value = {"valid": True, "message": "ok"}
    ws_factory = MagicMock(return_value=svc)
    monkeypatch.setattr(chat_api, "WebSearchService", ws_factory)

    resp = client.post("/api/chat/serp-status", json={"serp_api_key": "user-key"})
    assert resp.status_code == 200
    # The provided key must be forwarded to the service.
    ws_factory.assert_called_once_with(api_key="user-key")
    svc.validate_token.assert_called_once_with(api_key="user-key")


def test_serp_status_error_is_safe(monkeypatch):
    svc = MagicMock()
    svc.validate_token.side_effect = RuntimeError("network down")
    monkeypatch.setattr(chat_api, "WebSearchService", MagicMock(return_value=svc))

    resp = client.post("/api/chat/serp-status", json={})
    assert resp.status_code == 200
    # Errors are swallowed and reported as status "error", not a 500.
    assert resp.json()["status"] == "error"


# --------------------------------------------------------------------------- #
# Conversation memory endpoints
# --------------------------------------------------------------------------- #
def _fake_memory_service():
    svc = MagicMock()
    svc.get_session_stats.return_value = {
        "session_id": "s1",
        "message_count": 3,
        "total_tokens": 100,
        "summary_tokens": 0,
        "has_summary": False,
    }
    svc.max_context_tokens = 24_000
    svc.clear_session.return_value = True
    svc.summarize_if_needed = AsyncMock(return_value=True)
    return svc


def test_memory_stats(monkeypatch):
    svc = _fake_memory_service()
    monkeypatch.setattr(chat_api, "get_memory_service", MagicMock(return_value=svc))

    resp = client.get("/api/chat/memory/stats/s1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "s1"
    assert body["message_count"] == 3
    assert body["max_context_tokens"] == 24_000
    assert body["usage_percentage"] == pytest.approx(0.42, abs=0.01)


def test_memory_clear(monkeypatch):
    svc = _fake_memory_service()
    monkeypatch.setattr(chat_api, "get_memory_service", MagicMock(return_value=svc))

    resp = client.post("/api/chat/memory/clear/s1")
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
    svc.clear_session.assert_called_once_with("u1", "s1")


def test_memory_summarize(monkeypatch):
    svc = _fake_memory_service()
    monkeypatch.setattr(chat_api, "get_memory_service", MagicMock(return_value=svc))

    resp = client.post("/api/chat/memory/summarize/s1")
    assert resp.status_code == 200
    assert resp.json()["status"] == "summarized"
    svc.summarize_if_needed.assert_awaited_once()
