"""
Chat API endpoints with conversation memory support
"""
import logging
from typing import List, Dict
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

# Ensure environment variables are loaded before importing services
load_dotenv()

from app.backend.models.api_models import ChatRequestEnhanced, ChatResponse, SerpTokenStatus, MemoryStatsResponse
from app.services.ollama import stream_ollama
from app.services.vllm_service import vllm_service
from app.services.web_search_service import WebSearchService
from app.services.web_search_crew import WebSearchOrchestrator
from app.services.conversation_memory import get_memory_service, generate_session_id

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


def _build_messages_with_history(request: ChatRequestEnhanced) -> List[Dict[str, str]]:
    """
    Build the messages list including conversation history and summary.
    
    Args:
        request: The chat request with optional history/summary
        
    Returns:
        List of messages ready for LLM
    """
    messages = []
    
    # Add conversation summary as system context if available
    if request.conversation_summary:
        messages.append({
            "role": "system",
            "content": f"Summary of previous conversation:\n{request.conversation_summary}"
        })
    
    # Add conversation history if available
    if request.conversation_history:
        for msg in request.conversation_history:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
    
    # Add the current user message
    messages.append({"role": "user", "content": request.message})
    
    return messages


@router.get("/api/chat/memory/stats/{session_id}", response_model=MemoryStatsResponse)
async def get_memory_stats(session_id: str):
    """Get memory statistics for a session"""
    memory_service = get_memory_service()
    stats = memory_service.get_session_stats(session_id)
    
    return MemoryStatsResponse(
        session_id=stats["session_id"],
        message_count=stats["message_count"],
        total_tokens=stats["total_tokens"],
        summary_tokens=stats["summary_tokens"],
        has_summary=stats["has_summary"],
        max_context_tokens=memory_service.max_context_tokens,
        usage_percentage=round(
            (stats["total_tokens"] + stats["summary_tokens"]) / memory_service.max_context_tokens * 100, 2
        )
    )


@router.post("/api/chat/memory/clear/{session_id}")
async def clear_memory(session_id: str):
    """Clear memory for a session"""
    memory_service = get_memory_service()
    success = memory_service.clear_session(session_id)
    return {"status": "success" if success else "not_found", "session_id": session_id}


@router.post("/api/chat/memory/summarize/{session_id}")
async def summarize_memory(session_id: str):
    """Force summarization for a session"""
    memory_service = get_memory_service()
    summarized = await memory_service.summarize_if_needed(session_id, force=True)
    stats = memory_service.get_session_stats(session_id)
    return {
        "status": "summarized" if summarized else "no_action",
        "session_id": session_id,
        "stats": stats
    }


@router.post("/api/chat", response_model=ChatResponse)
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


@router.post("/api/chat/stream")
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
    """Handle vLLM chat request with conversation history"""
    if request.hf_token:
        token_result = vllm_service.set_hf_token(request.hf_token)
        if token_result["status"] == "error":
            raise HTTPException(status_code=400, detail=token_result["message"])
    
    if not await vllm_service.is_server_running():
        raise HTTPException(status_code=400, detail="vLLM server is not running")
    
    # Build messages with conversation history
    messages = _build_messages_with_history(request)
    
    response_text = ""
    async for chunk in vllm_service.stream_vllm_chat(
        messages=messages, 
        model=request.model, 
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_p=request.top_p,
        frequency_penalty=request.frequency_penalty,
        repetition_penalty=request.repetition_penalty
    ):
        response_text += chunk
    
    return ChatResponse(response=response_text, model=request.model)


async def _handle_ollama_chat(request: ChatRequestEnhanced) -> ChatResponse:
    """Handle Ollama chat request with conversation history"""
    # Build messages with conversation history
    messages = _build_messages_with_history(request)
    
    response_text = ""
    async for chunk in stream_ollama(
        messages, 
        model=request.model, 
        temperature=request.temperature, 
        max_tokens=request.max_tokens,
        top_p=request.top_p,
        frequency_penalty=request.frequency_penalty,
        repetition_penalty=request.repetition_penalty
    ):
        response_text += chunk
    
    return ChatResponse(response=response_text, model=request.model)


async def _stream_vllm_chat(request: ChatRequestEnhanced):
    """Stream vLLM chat responses with conversation history"""
    if request.hf_token:
        token_result = vllm_service.set_hf_token(request.hf_token)
        if token_result["status"] == "error":
            yield f"Error: {token_result['message']}"
            return
    
    if not await vllm_service.is_server_running():
        yield "Error: vLLM server is not running"
        return
    
    # Build messages with conversation history
    messages = _build_messages_with_history(request)
    
    async for chunk in vllm_service.stream_vllm_chat(
        messages=messages, 
        model=request.model, 
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_p=request.top_p,
        frequency_penalty=request.frequency_penalty,
        repetition_penalty=request.repetition_penalty
    ):
        yield chunk


async def _stream_ollama_chat(request: ChatRequestEnhanced):
    """Stream Ollama chat responses with conversation history"""
    # Build messages with conversation history
    messages = _build_messages_with_history(request)
    
    async for chunk in stream_ollama(
        messages, 
        model=request.model, 
        temperature=request.temperature, 
        max_tokens=request.max_tokens,
        top_p=request.top_p,
        frequency_penalty=request.frequency_penalty,
        repetition_penalty=request.repetition_penalty
    ):
        yield chunk


@router.post("/api/chat/web-search")
async def chat_web_search(request: ChatRequestEnhanced):
    """Process chat with optional web search integration"""
    logger.info(f"Web search chat request - use_web_search: {request.use_web_search}, Model: {request.model}")
    
    try:
        # Validate that web search is requested
        if not request.use_web_search:
            logger.info("Web search not requested, routing to standard chat endpoint")
            return await chat_stream(request)
        
        # Check if SerpAPI token is configured
        web_search_service = WebSearchService()
        token_validation = web_search_service.validate_token()
        
        if not token_validation["valid"]:
            logger.warning(f"SerpAPI token invalid: {token_validation['message']}")
            logger.info("Falling back to standard chat due to invalid token")
            
            # Return error message and fallback to standard chat
            async def error_and_fallback():
                error_msg = "SerpAPI token is required to perform Web search. Defaulting to local results.\n\n"
                yield error_msg
                
                # Stream standard chat response
                async for chunk in _stream_ollama_chat(request):
                    yield chunk
            
            return StreamingResponse(error_and_fallback(), media_type="text/plain")
        
        # Route to web search handler
        return await _handle_web_search_chat(request)
        
    except Exception as e:
        logger.error(f"Web search endpoint error: {e}", exc_info=True)
        logger.info("Falling back to standard chat due to error")
        
        # Fallback to standard chat on any error
        async def error_fallback():
            error_msg = f"Web search error: {str(e)}. Defaulting to local results.\n\n"
            yield error_msg
            
            async for chunk in _stream_ollama_chat(request):
                yield chunk
        
        return StreamingResponse(error_fallback(), media_type="text/plain")


@router.get("/api/chat/serp-status", response_model=SerpTokenStatus)
async def get_serp_token_status():
    """Get SerpAPI token validation status"""
    logger.info("Checking SerpAPI token status")
    
    try:
        web_search_service = WebSearchService()
        validation = web_search_service.validate_token()
        
        if validation["valid"]:
            return SerpTokenStatus(
                status="valid",
                message=validation["message"]
            )
        else:
            return SerpTokenStatus(
                status="invalid",
                message=validation["message"]
            )
            
    except Exception as e:
        logger.error(f"Error checking SerpAPI token status: {e}")
        return SerpTokenStatus(
            status="error",
            message=f"Error checking token status: {str(e)}"
        )


async def _handle_web_search_chat(request: ChatRequestEnhanced):
    """Handle web search chat request by routing to orchestrator"""
    logger.info("Routing to web search orchestrator")
    
    async def event_generator():
        try:
            # Initialize orchestrator
            orchestrator = WebSearchOrchestrator()
            
            # Build conversation history from request if available
            conversation_history = []
            # Note: ChatRequestEnhanced currently only has 'message', not full history
            # If history is needed, it would be added to the model
            
            # Process query through orchestrator with streaming
            async for chunk in orchestrator.process_query(
                query=request.message,
                conversation_history=conversation_history,
                stream=True
            ):
                yield chunk + "\n"
                
        except Exception as e:
            logger.error(f"Web search orchestrator error: {e}", exc_info=True)
            yield f"Error during web search: {str(e)}\n"
            yield "Falling back to local results...\n\n"
            
            # Fallback to standard chat
            async for chunk in _stream_ollama_chat(request):
                yield chunk + "\n"
    
    return StreamingResponse(event_generator(), media_type="text/plain")


async def _stream_web_search_chat(request: ChatRequestEnhanced):
    """Stream web search chat responses (helper for streaming)"""
    try:
        # Initialize orchestrator
        orchestrator = WebSearchOrchestrator()
        
        # Build conversation history from request if available
        conversation_history = []
        
        # Process query through orchestrator with streaming
        async for chunk in orchestrator.process_query(
            query=request.message,
            conversation_history=conversation_history,
            stream=True
        ):
            yield chunk
            
    except Exception as e:
        logger.error(f"Web search streaming error: {e}", exc_info=True)
        yield f"Error during web search: {str(e)}\n"
        yield "Falling back to local results...\n\n"
        
        # Fallback to standard chat
        async for chunk in _stream_ollama_chat(request):
            yield chunk
