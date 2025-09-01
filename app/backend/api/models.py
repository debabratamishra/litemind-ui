"""
Model and vLLM API endpoints
"""
import logging
from fastapi import APIRouter, HTTPException
import httpx

from app.backend.models.api_models import (
    ModelListResponse, VLLMTokenRequest, VLLMModelRequest, 
    VLLMStatusResponse, STTRequest, TranscriptionResponse
)
from app.backend.core.config import backend_config
from app.services.vllm_service import vllm_service
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


# vLLM endpoints
vllm_router = APIRouter(prefix="/api/vllm", tags=["vllm"])


@vllm_router.get("/models")
async def list_vllm_models():
    """Return available vLLM models.

    - local_models: Models detected in the local Hugging Face cache
    - popular_models: Kept for compatibility; currently empty as per UI change
    """
    try:
        data = vllm_service.get_available_models()
        # Drop popular models per request; keep key for compatibility
        return {
            "local_models": data.get("local_models", []),
            "popular_models": []
        }
    except Exception as e:
        logger.error(f"Failed to list vLLM models: {e}")
        return {"local_models": [], "popular_models": []}


@vllm_router.post("/set-token")
async def set_vllm_token(request: VLLMTokenRequest):
    """Set HuggingFace token for vLLM"""
    result = vllm_service.set_hf_token(request.token)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@vllm_router.post("/start-server")
async def start_vllm_server(request: VLLMModelRequest):
    """Start vLLM server with specified model"""
    kwargs = {}
    if request.dtype:
        kwargs["dtype"] = request.dtype
    if request.max_model_len:
        kwargs["max_model_len"] = request.max_model_len
    if request.gpu_memory_utilization:
        kwargs["gpu_memory_utilization"] = request.gpu_memory_utilization
    
    result = await vllm_service.start_vllm_server(request.model_name, **kwargs)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@vllm_router.post("/stop-server")
async def stop_vllm_server():
    """Stop vLLM server"""
    return await vllm_service.stop_vllm_server()


@vllm_router.post("/download-model")
async def download_vllm_model(request: VLLMModelRequest):
    """Download a model from Hugging Face Hub into the local cache."""
    try:
        result = await vllm_service.download_model(request.model_name)
        if result.get("status") != "success":
            raise HTTPException(status_code=400, detail=result.get("message", "Download failed"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@vllm_router.get("/server-status", response_model=VLLMStatusResponse)
async def vllm_server_status():
    """Check vLLM server status"""
    return VLLMStatusResponse(
        running=await vllm_service.is_server_running(),
        current_model=vllm_service.current_model
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


# Include vLLM and STT routers
router.include_router(vllm_router)
router.include_router(stt_router)
