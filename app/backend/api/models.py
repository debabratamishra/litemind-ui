"""Model and speech API endpoints."""
import logging
from fastapi import APIRouter, HTTPException
import httpx

from app.backend.models.api_models import (
    ModelListResponse, STTRequest, TranscriptionResponse,
    EnhancedModelListResponse, OllamaModelInfo,
)
from app.backend.core.config import backend_config
from app.backend.core.ollama_models import build_enhanced_model_payload
from app.services.speech_service import get_speech_service

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
            
    except Exception as e:
        logger.error(f"Failed to fetch models: {e}")
        raise HTTPException(status_code=500, detail=f"Could not fetch models: {str(e)}")


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
        transcription = speech_service.transcribe_audio(audio_bytes, request.sample_rate)
        
        return TranscriptionResponse(
            status="success" if transcription else "error",
            transcription=transcription or "",
            length=len(transcription) if transcription else 0
        )
        
    except Exception as e:
        logger.error(f"STT error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


# Include STT router
router.include_router(stt_router)
