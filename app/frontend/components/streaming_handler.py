"""
Streaming handlers for chat and RAG responses.
"""
import logging
import requests
from typing import Optional, Any

from .text_renderer import StreamingRenderer
from ..services.chat_service import chat_service
from ..services.rag_service import rag_service

logger = logging.getLogger(__name__)


class StreamingHandler:
    """Handles streaming responses from various backends."""
    
    def stream_chat_response(
        self,
        message: str,
        model: str,
        temperature: float = 0.7,
        backend: str = "ollama",
        hf_token: Optional[str] = None,
        placeholder: Optional[Any] = None,
        use_fastapi: bool = True
    ) -> Optional[str]:
        """Stream a chat response with reasoning segregation."""
        
        try:
            if not use_fastapi:
                # Local Ollama fallback
                return chat_service.stream_local_ollama_chat(
                    message=message,
                    model=model,
                    temperature=temperature
                )
            
            # FastAPI streaming
            response = chat_service.stream_fastapi_chat(
                message=message,
                model=model,
                temperature=temperature,
                backend=backend,
                hf_token=hf_token
            )
            
            return self._process_streaming_response(response, placeholder)
            
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
        placeholder: Optional[Any] = None
    ) -> Optional[str]:
        """Stream a RAG response with reasoning segregation."""
        
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
                hf_token=hf_token
            )
            
            return self._process_streaming_response(response, placeholder)
            
        except requests.Timeout:
            logger.error("RAG API timed out while streaming.")
            if placeholder:
                placeholder.error("❌ RAG API timed out while streaming. Try narrowing the query.")
        except requests.RequestException as exc:
            logger.error(f"RAG API Error: {exc}")
            if placeholder:
                placeholder.error(f"❌ RAG API Error: {exc}")
        
        return None
    
    def _process_streaming_response(
        self, 
        response: requests.Response, 
        placeholder: Optional[Any]
    ) -> str:
        """Process streaming response and handle reasoning segregation."""
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

        return buf


# Global streaming handler
streaming_handler = StreamingHandler()
