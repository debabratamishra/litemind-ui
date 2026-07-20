"""Unit tests for ``app.services.ollama`` (offline).

This module is a thin wrapper: ``get_ollama_url`` delegates to the gateway's
``get_ollama_api_base`` (which resolves via ``host_service_manager`` in this
environment), and ``stream_ollama`` forwards to ``stream_completion`` with
``backend="ollama"``. We mock the gateway boundary so no real network call is
made.
"""
from unittest.mock import patch

from app.services import llm_gateway as gw
from app.services import ollama


def test_get_ollama_url_returns_http_url():
    url = ollama.get_ollama_url()
    assert url.startswith("http")


def test_get_ollama_url_delegates_to_api_base():
    with patch.object(ollama, "get_ollama_api_base", return_value="http://custom:11434"):
        assert ollama.get_ollama_url() == "http://custom:11434"


async def test_stream_ollama_streams_via_gateway(monkeypatch):
    # Drive the real stream_completion -> _stream_ollama_native path with the
    # native client stubbed out, proving stream_ollama forwards through the gateway.
    async def fake_native(*a, **k):
        yield "hi"
        yield " there"

    monkeypatch.setattr(gw, "_stream_ollama_native", fake_native)

    chunks = [
        c
        async for c in ollama.stream_ollama(
            messages=[{"role": "user", "content": "hello"}], model="llama3"
        )
    ]
    assert "".join(chunks) == "hi there"


async def test_stream_ollama_passes_through_parameters(monkeypatch):
    captured = {}

    async def fake_native(*a, **k):
        captured.update(k)
        yield "x"

    monkeypatch.setattr(gw, "_stream_ollama_native", fake_native)

    chunks = [
        c
        async for c in ollama.stream_ollama(
            messages=[],
            model="mistral",
            temperature=0.3,
            max_tokens=512,
            top_p=0.8,
            frequency_penalty=0.1,
            repetition_penalty=1.2,
        )
    ]
    assert chunks == ["x"]
    assert captured["model"] == "ollama_chat/mistral"
    assert captured["temperature"] == 0.3
    assert captured["max_tokens"] == 512
    assert captured["top_p"] == 0.8
    assert captured["frequency_penalty"] == 0.1
    assert captured["repetition_penalty"] == 1.2
