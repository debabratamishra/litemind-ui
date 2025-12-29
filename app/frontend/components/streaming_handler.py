"""
Streaming handlers for chat and RAG responses with conversation memory support.
"""
import logging
import requests
from typing import Optional, Any, Callable, List, Dict

from .text_renderer import StreamingRenderer, plain_text_renderer, web_search_renderer
from ..services.chat_service import chat_service
from ..services.rag_service import rag_service
from ..utils.text_processing import normalize_plain_text_spacing, format_web_search_response

logger = logging.getLogger(__name__)


class StreamingHandler:
    """Handles streaming responses from various backends with conversation memory."""
    
    def stream_chat_response(
        self,
        message: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        backend: str = "ollama",
        hf_token: Optional[str] = None,
        placeholder: Optional[Any] = None,
        use_fastapi: bool = True,
        tts_callback: Optional[Callable[[str], None]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        conversation_summary: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Optional[str]:
        """Stream a chat response with reasoning segregation and conversation memory.
        
        Args:
            message: User message
            model: Model name
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            backend: Backend to use (ollama/vllm)
            hf_token: HuggingFace token for vLLM
            placeholder: Streamlit placeholder for UI updates
            use_fastapi: Whether to use FastAPI backend
            tts_callback: Optional callback to receive text chunks for TTS synthesis.
                         Called with each text chunk as it arrives for real-time TTS.
            conversation_history: Previous messages in the conversation
            conversation_summary: Summary of earlier messages
            session_id: Session identifier for memory tracking
        """
        
        try:
            if not use_fastapi:
                # Local Ollama fallback with memory
                return chat_service.stream_local_ollama_chat(
                    message=message,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    conversation_history=conversation_history,
                    conversation_summary=conversation_summary
                )
            
            # FastAPI streaming with memory
            response = chat_service.stream_fastapi_chat(
                message=message,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                backend=backend,
                hf_token=hf_token,
                conversation_history=conversation_history,
                conversation_summary=conversation_summary,
                session_id=session_id
            )
            
            return self._process_streaming_response(response, placeholder, tts_callback=tts_callback)
            
        except requests.Timeout:
            logger.error("Chat API timed out while streaming.")
            if placeholder:
                placeholder.error("❌ Chat API timed out while streaming.")
        except requests.RequestException as exc:
            logger.error(f"Chat API Error: {exc}")
            if placeholder:
                placeholder.error(f"❌ Chat API Error: {exc}")
        
        return None
    
    def stream_rag_response(
        self,
        query: str,
        messages: list,
        model: str,
        system_prompt: str,
        n_results: int = 3,
        use_multi_agent: bool = False,
        use_hybrid_search: bool = False,
        backend: str = "ollama",
        hf_token: Optional[str] = None,
        placeholder: Optional[Any] = None,
        tts_callback: Optional[Callable[[str], None]] = None,
        conversation_summary: Optional[str] = None,
        session_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> Optional[str]:
        """Stream a RAG response with reasoning segregation and conversation memory.
        
        Args:
            query: User query
            messages: Chat history
            model: Model name
            system_prompt: System prompt for RAG
            n_results: Number of results to retrieve
            use_multi_agent: Whether to use multi-agent processing
            use_hybrid_search: Whether to use hybrid search
            backend: Backend to use (ollama/vllm)
            hf_token: HuggingFace token for vLLM
            placeholder: Streamlit placeholder for UI updates
            tts_callback: Optional callback to receive text chunks for TTS synthesis
            conversation_summary: Summary of earlier messages
            session_id: Session identifier for memory tracking
            temperature: Temperature for LLM response generation
            max_tokens: Maximum tokens to generate
        """
        
        try:
            response = rag_service.stream_rag_query(
                query=query,
                messages=messages,
                model=model,
                system_prompt=system_prompt,
                n_results=n_results,
                use_multi_agent=use_multi_agent,
                use_hybrid_search=use_hybrid_search,
                backend=backend,
                hf_token=hf_token,
                conversation_summary=conversation_summary,
                session_id=session_id,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return self._process_streaming_response(response, placeholder, tts_callback=tts_callback)
            
        except requests.Timeout:
            logger.error("RAG API timed out while streaming.")
            if placeholder:
                placeholder.error("❌ RAG API timed out while streaming. Try narrowing the query.")
        except requests.RequestException as exc:
            logger.error(f"RAG API Error: {exc}")
            if placeholder:
                placeholder.error(f"❌ RAG API Error: {exc}")
        
        return None
    
    def stream_web_search_response(
        self,
        message: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        backend: str = "ollama",
        hf_token: Optional[str] = None,
        placeholder: Optional[Any] = None,
        use_fastapi: bool = True,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        conversation_summary: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Optional[str]:
        """Stream a web search response with raw LLM output (no formatting) and memory."""
        
        try:
            if not use_fastapi:
                # Fallback to local Ollama chat if FastAPI not available
                logger.warning("Web search requires FastAPI backend. Falling back to local chat.")
                return chat_service.stream_local_ollama_chat(
                    message=message,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    conversation_history=conversation_history,
                    conversation_summary=conversation_summary
                )
            
            # FastAPI web search streaming with memory
            response = chat_service.stream_web_search_chat(
                message=message,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                backend=backend,
                hf_token=hf_token,
                conversation_history=conversation_history,
                conversation_summary=conversation_summary,
                session_id=session_id
            )
            
            # Use raw text processing for web search (no formatting)
            return self._process_streaming_response_raw(response, placeholder)
            
        except requests.Timeout:
            logger.error("Web search API timed out while streaming.")
            if placeholder:
                placeholder.error("❌ Web search API timed out. Falling back to standard chat.")
        except requests.RequestException as exc:
            logger.error(f"Web search API Error: {exc}")
            if placeholder:
                placeholder.error(f"❌ Web search API Error: {exc}")
        
        return None
    
    def _process_streaming_response(
        self, 
        response: requests.Response, 
        placeholder: Optional[Any],
        tts_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """Process streaming response and handle reasoning segregation.
        
        Args:
            response: Streaming HTTP response
            placeholder: Streamlit placeholder for UI updates
            tts_callback: Optional callback for TTS chunks (called for each text chunk)
        """
        buf = ""
        segregator = StreamingRenderer(placeholder) if placeholder is not None else None

        for line in response.iter_lines(decode_unicode=True, chunk_size=1):
            if not line:
                continue
            if line.startswith("data:"):
                payload = line[5:].lstrip()
                if payload.strip() in ("[DONE]", ""):
                    continue
                line = payload

            chunk = line + "\n"
            buf += chunk
            if segregator is not None:
                segregator.feed(chunk)
            
            # Call TTS callback with the text chunk for streaming synthesis
            if tts_callback is not None:
                try:
                    tts_callback(chunk)
                except Exception as e:
                    logger.debug(f"TTS callback error: {e}")

        return buf
    
    def _process_streaming_response_raw(
        self, 
        response: requests.Response, 
        placeholder: Optional[Any]
    ) -> str:
        """Process streaming response for web search with proper formatting."""
        # Reset the web search renderer for new response
        web_search_renderer.reset()
        
        for line in response.iter_lines(decode_unicode=True, chunk_size=1):
            if line is None:
                continue

            if line.startswith("data:"):
                payload = line[5:].lstrip()
                if payload.strip() in ("[DONE]", ""):
                    continue
                line = payload

            # Handle empty lines as newlines
            if line == "":
                chunk = "\n"
            else:
                chunk = line

            # Use the web search renderer for streaming
            if placeholder is not None:
                web_search_renderer.render_streaming(chunk, placeholder=placeholder)
            else:
                web_search_renderer.render_streaming(chunk, placeholder=None)

        # Return the final formatted text
        return web_search_renderer.get_final_text()


# Global streaming handler
streaming_handler = StreamingHandler()
