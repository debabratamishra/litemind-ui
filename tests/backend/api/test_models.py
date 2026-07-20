"""Unit tests for ``app.backend.api.models`` (offline).

``get_available_models`` performs a real HTTP GET to Ollama; we route it through
the shared ``httpx_mock`` transport so no network call escapes. ``get_enhanced_models``
delegates to ``build_enhanced_model_payload`` (stubbed), and ``transcribe_audio``
delegates to the speech service (stubbed).
"""
import base64
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException

from app.backend.api import models
from app.backend.models.api_models import (
    EnhancedModelListResponse,
    ModelListResponse,
    OllamaModelInfo,
    STTRequest,
    TranscriptionResponse,
)


async def test_get_available_models(httpx_mock):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "llama3:latest"}, {"name": "gemma3:1b"}]})

    httpx_mock(handler)
    result = await models.get_available_models()
    assert isinstance(result, ModelListResponse)
    assert result.models == ["llama3:latest", "gemma3:1b"]


async def test_get_available_models_empty(httpx_mock):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": []})

    httpx_mock(handler)
    result = await models.get_available_models()
    assert result.models == []


async def test_get_available_models_provider_error(httpx_mock):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    httpx_mock(handler)
    with pytest.raises(HTTPException) as excinfo:
        await models.get_available_models()
    assert excinfo.value.status_code == 500
    assert "Could not fetch models" in excinfo.value.detail


async def test_get_enhanced_models():
    payload = {
        "local_models": [
            {"name": "llama3:latest", "parameter_size": "8B", "is_local": True}
        ],
        "cloud_models": [
            {
                "name": "gemma4:31b-cloud",
                "parameter_size": "31B",
                "family": "gemma4",
                "is_local": False,
                "description": "Google Gemma 4",
            }
        ],
    }
    with patch.object(
        models, "build_enhanced_model_payload", new=AsyncMock(return_value=payload)
    ):
        result = await models.get_enhanced_models()

    assert isinstance(result, EnhancedModelListResponse)
    assert len(result.local_models) == 1
    assert isinstance(result.local_models[0], OllamaModelInfo)
    assert result.local_models[0].name == "llama3:latest"
    assert result.local_models[0].is_local is True
    assert len(result.cloud_models) == 1
    assert result.cloud_models[0].is_local is False


async def test_transcribe_audio_success():
    service = MagicMock()
    service.transcribe_audio.return_value = "hello world"

    audio = base64.b64encode(b"fake pcm bytes").decode()
    request = STTRequest(audio_data=audio, sample_rate=16000)

    with patch.object(models, "get_speech_service", return_value=service):
        result = await models.transcribe_audio(request)

    assert isinstance(result, TranscriptionResponse)
    assert result.status == "success"
    assert result.transcription == "hello world"
    assert result.length == len("hello world")
    service.transcribe_audio.assert_called_once()


async def test_transcribe_audio_empty_transcript():
    # Empty transcript string -> status "error", length 0 (no exception).
    service = MagicMock()
    service.transcribe_audio.return_value = ""

    audio = base64.b64encode(b"silent").decode()
    request = STTRequest(audio_data=audio)

    with patch.object(models, "get_speech_service", return_value=service):
        result = await models.transcribe_audio(request)

    assert result.status == "error"
    assert result.transcription == ""
    assert result.length == 0


async def test_transcribe_audio_service_error():
    service = MagicMock()
    service.transcribe_audio.side_effect = RuntimeError("model missing")

    audio = base64.b64encode(b"data").decode()
    request = STTRequest(audio_data=audio)

    with patch.object(models, "get_speech_service", return_value=service):
        with pytest.raises(HTTPException) as excinfo:
            await models.transcribe_audio(request)
    assert excinfo.value.status_code == 500
    assert "Transcription failed" in excinfo.value.detail
