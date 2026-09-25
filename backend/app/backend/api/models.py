"""Model and speech API endpoints."""
import logging

import httpx
from fastapi import APIRouter, HTTPException

from backend.app.backend.core.config import backend_config
from backend.app.backend.core.ollama_models import build_enhanced_model_payload
from backend.app.backend.models.api_models import (
    EnhancedModelListResponse,
    ModelListResponse,
    OllamaModelInfo,
    STTRequest,
    TranscriptionResponse,
)
from backend.app.services.speech_service import get_speech_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["models"])


@router.get("/models", response_model=ModelListResponse)
async def get_available_models():
    """Get available Ollama models"""
    try:
        ollama_url = backend_config.get_ollama_url()

        async with httpx.AsyncClient() as client:
            response = await client.get(f"{ollama_url}/api/tags")
            response.raise_for_status()
            data = response.json()

            models = [model["name"] for model in data.get("models", [])]
            return ModelListResponse(models=models)

    except Exception:
        logger.exception("Failed to fetch models")
        raise HTTPException(status_code=500, detail="Could not fetch models")


@router.get("/models/enhanced", response_model=EnhancedModelListResponse)
async def get_enhanced_models():
    """Return local models with metadata + cloud catalog with availability flags."""
    ollama_url = backend_config.get_ollama_url()
    payload = await build_enhanced_model_payload(ollama_url)
    return EnhancedModelListResponse(
        local_models=[OllamaModelInfo(**model) for model in payload["local_models"]],
        cloud_models=[OllamaModelInfo(**model) for model in payload["cloud_models"]],
    )


# Speech-to-Text endpoints
stt_router = APIRouter(prefix="/api/stt", tags=["speech"])


@stt_router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(request: STTRequest):
    """Transcribe audio data"""
    try:
        import base64

        audio_bytes = base64.b64decode(request.audio_data)
        speech_service = get_speech_service()
        transcription = speech_service.transcribe_audio(audio_bytes, request.sample_rate or 16000)

        return TranscriptionResponse(
            status="success" if transcription else "error",
            transcription=transcription or "",
            length=len(transcription) if transcription else 0
        )

    except Exception:
        logger.exception("STT transcription error")
        raise HTTPException(status_code=500, detail="Transcription failed")


@stt_router.get("/status")
async def get_stt_status():
    """Get STT service status."""
    try:
        speech_service = get_speech_service()
        return speech_service.get_status()
    except Exception:
        logger.exception("Failed to get STT status")
        return {"available": False, "error": "Unable to retrieve STT status"}


# Include STT router
router.include_router(stt_router)
