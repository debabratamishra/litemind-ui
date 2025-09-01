"""
Chat API endpoints
"""
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.backend.models.api_models import ChatRequestEnhanced, ChatResponse
from app.services.ollama import stream_ollama
from app.services.vllm_service import vllm_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequestEnhanced):
    """Single chat message processing"""
    logger.info(f"Chat request - Backend: {request.backend}, Model: {request.model}")
    
    try:
        if request.backend == "vllm":
            return await _handle_vllm_chat(request)
        else:
            return await _handle_ollama_chat(request)
            
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def chat_stream(request: ChatRequestEnhanced):
    """Stream chat responses"""
    logger.info(f"Streaming chat - Backend: {request.backend}, Model: {request.model}")
    
    async def event_generator():
        try:
            if request.backend == "vllm":
                async for chunk in _stream_vllm_chat(request):
                    yield chunk + "\n"
            else:
                async for chunk in _stream_ollama_chat(request):
                    yield chunk + "\n"
                    
        except Exception as e:
            logger.error(f"Chat streaming error: {e}")
            yield f"Error: {str(e)}\n"

    return StreamingResponse(event_generator(), media_type="text/plain")


async def _handle_vllm_chat(request: ChatRequestEnhanced) -> ChatResponse:
    """Handle vLLM chat request"""
    if request.hf_token:
        token_result = vllm_service.set_hf_token(request.hf_token)
        if token_result["status"] == "error":
            raise HTTPException(status_code=400, detail=token_result["message"])
    
    if not await vllm_service.is_server_running():
        raise HTTPException(status_code=400, detail="vLLM server is not running")
    
    messages = [{"role": "user", "content": request.message}]
    response_text = ""
    async for chunk in vllm_service.stream_vllm_chat(
        messages=messages, model=request.model, temperature=request.temperature
    ):
        response_text += chunk
    
    return ChatResponse(response=response_text, model=request.model)


async def _handle_ollama_chat(request: ChatRequestEnhanced) -> ChatResponse:
    """Handle Ollama chat request"""
    messages = [{"role": "user", "content": request.message}]
    response_text = ""
    async for chunk in stream_ollama(messages, model=request.model, temperature=request.temperature):
        response_text += chunk
    
    return ChatResponse(response=response_text, model=request.model)


async def _stream_vllm_chat(request: ChatRequestEnhanced):
    """Stream vLLM chat responses"""
    if request.hf_token:
        token_result = vllm_service.set_hf_token(request.hf_token)
        if token_result["status"] == "error":
            yield f"Error: {token_result['message']}"
            return
    
    if not await vllm_service.is_server_running():
        yield "Error: vLLM server is not running"
        return
    
    messages = [{"role": "user", "content": request.message}]
    async for chunk in vllm_service.stream_vllm_chat(
        messages=messages, model=request.model, temperature=request.temperature
    ):
        yield chunk


async def _stream_ollama_chat(request: ChatRequestEnhanced):
    """Stream Ollama chat responses"""
    messages = [{"role": "user", "content": request.message}]
    async for chunk in stream_ollama(messages, model=request.model, temperature=request.temperature):
        yield chunk
