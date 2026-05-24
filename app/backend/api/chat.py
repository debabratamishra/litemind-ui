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
import json

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

# Minimum output-token budget enforced when Generative UI is active.
# Complex artifacts (canvas games, dashboards) can easily need 8 000–12 000
# tokens; 16 384 provides comfortable headroom without hitting most model
# context-window limits.
GENUI_MIN_MAX_TOKENS = 16_384


GENERATIVE_UI_SYSTEM_PROMPT = (
    "You can embed rich UI components in your responses using fenced code "
    "blocks with a ui: language tag followed by a JSON body on the NEXT line.\n\n"
    "FORMATTING RULES:\n"
    "1. Start the block with ```ui:component_name\n"
    "2. Put valid JSON props on the next line(s)\n"
    "3. Close with ``` on its own line\n"
    "4. Mix normal markdown text freely between component blocks\n"
    "5. Exceptions: ```ui:webapp and ```ui:iframe_app use raw HTML/CSS/JS instead of JSON\n\n"
    "EXAMPLE – comparison table:\n"
    "```ui:data_table\n"
    '{"title": "Model Comparison", "columns": ["Model", "Size", "Speed"], '
    '"data": [["Gemma", "1B", "Fast"], ["Llama", "8B", "Medium"]]}\n'
    "```\n\n"
    "EXAMPLE – interactive webapp:\n"
    "```ui:webapp\n"
    "<!-- height: 620 -->\n"
    "<div class=\"toy-counter\">\n"
    "  <h3>Toy Counter</h3>\n"
    "  <p id=\"count\">0</p>\n"
    "  <button id=\"increment\">Increment</button>\n"
    "  <button id=\"reset\">Reset</button>\n"
    "</div>\n"
    "<style>\n"
    "  .toy-counter { display: grid; gap: 12px; justify-items: start; }\n"
    "  .toy-counter button { border: 0; border-radius: 999px; padding: 10px 16px; cursor: pointer; }\n"
    "</style>\n"
    "<script>\n"
    "  const countEl = document.getElementById(\"count\");\n"
    "  let count = 0;\n"
    "  document.getElementById(\"increment\").addEventListener(\"click\", () => {\n"
    "    count += 1;\n"
    "    countEl.textContent = String(count);\n"
    "  });\n"
    "  document.getElementById(\"reset\").addEventListener(\"click\", () => {\n"
    "    count = 0;\n"
    "    countEl.textContent = \"0\";\n"
    "  });\n"
    "</script>\n"
    "```\n\n"
    "EXAMPLE – playable iframe app:\n"
    "```ui:iframe_app\n"
    "<!-- height: 720 -->\n"
    "<div id=\"app\" data-autofocus>\n"
    "  <canvas id=\"maze\" width=\"480\" height=\"480\" aria-label=\"Arcade demo\"></canvas>\n"
    "  <p>Use the arrow keys to move.</p>\n"
    "</div>\n"
    "<style>\n"
    "  #app { display: grid; place-items: center; gap: 12px; padding: 16px; background: #050816; color: #f8fafc; }\n"
    "  canvas { background: #111827; border-radius: 16px; }\n"
    "</style>\n"
    "<script>\n"
    "  const canvas = document.getElementById(\"maze\");\n"
    "  const ctx = canvas.getContext(\"2d\");\n"
    "  let x = 40;\n"
    "  window.addEventListener(\"keydown\", (event) => {\n"
    "    if (event.key === \"ArrowRight\") x = Math.min(x + 16, canvas.width - 24);\n"
    "    if (event.key === \"ArrowLeft\") x = Math.max(x - 16, 8);\n"
    "  });\n"
    "  function draw() {\n"
    "    ctx.clearRect(0, 0, canvas.width, canvas.height);\n"
    "    ctx.fillStyle = \"#facc15\";\n"
    "    ctx.beginPath();\n"
    "    ctx.arc(x, 240, 18, 0.25 * Math.PI, 1.75 * Math.PI);\n"
    "    ctx.lineTo(x, 240);\n"
    "    ctx.fill();\n"
    "    requestAnimationFrame(draw);\n"
    "  }\n"
    "  draw();\n"
    "</script>\n"
    "```\n\n"
    "EXAMPLE – key metrics side by side:\n"
    "```ui:metric\n"
    '{"metrics": [{"label": "Users", "value": "1,234", "delta": "+12%"}, '
    '{"label": "Latency", "value": "45ms", "delta": "-5ms"}]}\n'
    "```\n\n"
    "EXAMPLE – bar chart:\n"
    "```ui:chart\n"
    '{"type": "bar", "title": "Quarterly Sales", '
    '"x": ["Q1", "Q2", "Q3", "Q4"], "y": [100, 150, 200, 180]}\n'
    "```\n\n"
    "EXAMPLE – info card:\n"
    "```ui:info_card\n"
    '{"icon": "💡", "title": "Tip", "content": "Use streaming for faster responses", "color": "#4CAF50"}\n'
    "```\n\n"
    "Available components: data_table (columns + data), metric (label/value/delta), "
    "chart (type: bar/line/pie/scatter, x, y), info_card (icon/title/content/color), "
    "webapp (self-contained HTML/CSS/JS mini-apps with inline styles/scripts), "
    "iframe_app (self-contained HTML/CSS/JS apps and games rendered in an iframe), "
    "button_group (label/buttons with text/value), "
    "alert (level: info/success/warning/error, message), "
    "steps (steps/current), tabs (tabs with label/content), "
    "callout (emoji/title/content), columns (items with title/content/icon), "
    "json_viewer (title/data), progress (value/label), "
    "link_cards (links with title/url/description).\n\n"
    "WHEN TO USE:\n"
    "- Comparing items → data_table\n"
    "- Key numbers/statistics → metric\n"
    "- Trends over time → chart\n"
    "- Step-by-step instructions → steps\n"
    "- Important notices → alert or callout\n"
    "- Lightweight interactive snippets → webapp\n"
    "- Playable games, dashboards, calculators, editors, and simulations → iframe_app\n"
    "- Simple text answers → just use normal markdown, no components needed\n\n"
    "EMBEDDED APP RULES:\n"
    "- Keep webapps and iframe apps self-contained with inline HTML/CSS/JS\n"
    "- Do not rely on external CDNs, npm packages, or build steps\n"
    "- Ensure buttons, animations, and controls work with client-side JavaScript only\n"
    "- Never return bare HTML or ```html fences when Generative UI is enabled; always wrap app output in ```ui:webapp or ```ui:iframe_app\n"
    "- Use ```ui:iframe_app when the user explicitly asks for an app, game, simulator, or playground inside the chat\n"
    "- Put any explanation outside the ui block; do not append notes after </html> inside the block\n"
    "- Make the first frame visibly non-empty even before JavaScript runs: include a heading, button, canvas, text, or loading label in the HTML body\n"
    "- Avoid full-screen blank black/white backgrounds with no visible text or controls\n"
    "- Include a visible control hint for keyboard-driven apps\n"
    "- Prefer an explicit <!-- height: N --> comment for taller apps\n"
    "- Return one complete runnable block for the app itself; keep any explanation outside the block\n\n"
    "FALLBACK: If you are unsure about the component JSON syntax, use standard "
    "markdown tables and **Bold Label:** Value lines instead – those will be "
    "auto-converted to rich components."
)


def _build_messages_with_history(request: ChatRequestEnhanced) -> List[Dict[str, str]]:
    """
    Build the messages list including conversation history and summary.
    Also applies voice mode optimizations if is_voice_mode is True.
    
    Args:
        request: The chat request with optional history/summary
        
    Returns:
        List of messages ready for LLM
    """
    messages = []
    
    # Add voice mode system prompt if voice mode is active
    if request.is_voice_mode:
        from app.frontend.config import DEFAULT_CHAT_SYSTEM_PROMPT_VOICE
        messages.append({
            "role": "system",
            "content": DEFAULT_CHAT_SYSTEM_PROMPT_VOICE
        })

    if request.enable_generative_ui:
        messages.append({
            "role": "system",
            "content": GENERATIVE_UI_SYSTEM_PROMPT
        })
    
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
    
    # Add the current user message (with optional generative UI hint)
    user_content = request.message
    if request.enable_generative_ui:
        user_content += (
            "\n\n[Respond using rich UI components when helpful: "
            "```ui:data_table for comparisons/tables, "
            "```ui:metric for key numbers, "
            "```ui:chart for trends. "
            "```ui:webapp for lightweight interactive snippets. "
            "```ui:iframe_app for runnable apps, games, dashboards, editors, and simulations inside the chat iframe. "
            "If unsure about component syntax, use standard markdown tables "
            "and **Bold Label:** Value lines instead.]"
        )
    messages.append({"role": "user", "content": user_content})
    
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
                    payload = json.dumps({"chunk": chunk}, ensure_ascii=False)
                    yield f"data: {payload}\n\n"
            else:
                async for chunk in _stream_ollama_chat(request):
                    payload = json.dumps({"chunk": chunk}, ensure_ascii=False)
                    yield f"data: {payload}\n\n"
                    
        except Exception as e:
            logger.error(f"Chat streaming error: {e}")
            payload = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


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
    
    # Adjust max_tokens: voice mode → short; GenUI mode → at least GENUI_MIN_MAX_TOKENS
    if request.is_voice_mode:
        max_tokens = 300
    elif request.enable_generative_ui:
        max_tokens = max(request.max_tokens or 2048, GENUI_MIN_MAX_TOKENS)
    else:
        max_tokens = request.max_tokens
    
    response_text = ""
    async for chunk in vllm_service.stream_vllm_chat(
        messages=messages, 
        model=request.model, 
        temperature=request.temperature,
        max_tokens=max_tokens,
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
    
    # Adjust max_tokens: voice mode → short; GenUI mode → at least GENUI_MIN_MAX_TOKENS
    if request.is_voice_mode:
        max_tokens = 300
    elif request.enable_generative_ui:
        max_tokens = max(request.max_tokens or 2048, GENUI_MIN_MAX_TOKENS)
    else:
        max_tokens = request.max_tokens
    
    response_text = ""
    async for chunk in stream_ollama(
        messages, 
        model=request.model, 
        temperature=request.temperature, 
        max_tokens=max_tokens,
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
    
    # Adjust max_tokens: voice mode → short; GenUI mode → at least GENUI_MIN_MAX_TOKENS
    if request.is_voice_mode:
        max_tokens = 300
    elif request.enable_generative_ui:
        max_tokens = max(request.max_tokens or 2048, GENUI_MIN_MAX_TOKENS)
    else:
        max_tokens = request.max_tokens
    
    async for chunk in vllm_service.stream_vllm_chat(
        messages=messages, 
        model=request.model, 
        temperature=request.temperature,
        max_tokens=max_tokens,
        top_p=request.top_p,
        frequency_penalty=request.frequency_penalty,
        repetition_penalty=request.repetition_penalty
    ):
        yield chunk


async def _stream_ollama_chat(request: ChatRequestEnhanced):
    """Stream Ollama chat responses with conversation history"""
    # Build messages with conversation history
    messages = _build_messages_with_history(request)
    
    # Adjust max_tokens: voice mode → short; GenUI mode → at least GENUI_MIN_MAX_TOKENS
    if request.is_voice_mode:
        max_tokens = 300
    elif request.enable_generative_ui:
        max_tokens = max(request.max_tokens or 2048, GENUI_MIN_MAX_TOKENS)
    else:
        max_tokens = request.max_tokens
    
    async for chunk in stream_ollama(
        messages, 
        model=request.model, 
        temperature=request.temperature, 
        max_tokens=max_tokens,
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
