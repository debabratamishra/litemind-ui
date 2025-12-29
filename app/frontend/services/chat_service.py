"""
Chat service for handling chat interactions and streaming with conversation memory.
"""
import asyncio
import logging
import requests
from typing import Dict, List, Optional, Any

from ..config import FASTAPI_URL, CONNECT_TIMEOUT, READ_TIMEOUT
from ...services.ollama import stream_ollama

logger = logging.getLogger(__name__)


class ChatService:
    """Service for handling chat interactions with conversation memory support."""
    
    def __init__(self):
        self.base_url = FASTAPI_URL

    def call_fastapi_chat(
        self, 
        message: str, 
        model: str = "default", 
        temperature: float = 0.7,
        max_tokens: int = 2048,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        conversation_summary: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Optional[str]:
        """Call the non-streaming chat endpoint with conversation memory."""
        try:
            payload = {
                "message": message, 
                "model": model, 
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Add conversation history if provided
            if conversation_history:
                payload["conversation_history"] = conversation_history
            if conversation_summary:
                payload["conversation_summary"] = conversation_summary
            if session_id:
                payload["session_id"] = session_id
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120,
            )
            if response.status_code == 200:
                return response.json().get("response")
            logger.error(f"FastAPI Error: {response.status_code}")
            return None
        except requests.RequestException as exc:
            logger.error(f"FastAPI Connection Error: {exc}")
            return None

    def stream_fastapi_chat(
        self,
        message: str,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        backend: str = "ollama",
        hf_token: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        conversation_summary: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> requests.Response:
        """Stream a chat response from the backend with conversation memory."""
        payload = {
            "message": message,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "backend": backend
        }
        
        if backend == "vllm" and hf_token:
            payload["hf_token"] = hf_token
        
        # Add conversation history if provided
        if conversation_history:
            payload["conversation_history"] = conversation_history
        if conversation_summary:
            payload["conversation_summary"] = conversation_summary
        if session_id:
            payload["session_id"] = session_id
        
        response = requests.post(
            f"{self.base_url}/api/chat/stream",
            json=payload,
            stream=True,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )
        response.raise_for_status()
        return response

    def stream_local_ollama_chat(
        self,
        message: str,
        model: str,
        temperature: float,
        max_tokens: int = 2048,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        conversation_summary: Optional[str] = None
    ) -> str:
        """Stream a chat response from local Ollama with conversation memory."""
        async def _inner() -> str:
            # Build messages with history
            messages = []
            
            # Add summary as system context
            if conversation_summary:
                messages.append({
                    "role": "system",
                    "content": f"Summary of previous conversation:\n{conversation_summary}"
                })
            
            # Add conversation history
            if conversation_history:
                messages.extend(conversation_history)
            
            # Add current message
            messages.append({"role": "user", "content": message})
            
            acc = ""
            async for chunk in stream_ollama(
                messages, 
                model=model, 
                temperature=temperature,
                max_tokens=max_tokens
            ):
                acc += chunk
            return acc

        return asyncio.run(_inner())
    
    def stream_web_search_chat(
        self,
        message: str,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        backend: str = "ollama",
        hf_token: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        conversation_summary: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> requests.Response:
        """Stream a web search chat response from the backend with conversation memory."""
        payload = {
            "message": message,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "backend": backend,
            "use_web_search": True
        }
        
        if backend == "vllm" and hf_token:
            payload["hf_token"] = hf_token
        
        # Add conversation history if provided
        if conversation_history:
            payload["conversation_history"] = conversation_history
        if conversation_summary:
            payload["conversation_summary"] = conversation_summary
        if session_id:
            payload["session_id"] = session_id
        
        response = requests.post(
            f"{self.base_url}/api/chat/web-search",
            json=payload,
            stream=True,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )
        response.raise_for_status()
        return response

    def get_memory_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get memory statistics for a session."""
        try:
            response = requests.get(
                f"{self.base_url}/api/chat/memory/stats/{session_id}",
                timeout=10,
            )
            if response.status_code == 200:
                return response.json()
            return None
        except requests.RequestException as exc:
            logger.error(f"Memory stats error: {exc}")
            return None

    def clear_memory(self, session_id: str) -> bool:
        """Clear memory for a session."""
        try:
            response = requests.post(
                f"{self.base_url}/api/chat/memory/clear/{session_id}",
                timeout=10,
            )
            return response.status_code == 200
        except requests.RequestException as exc:
            logger.error(f"Memory clear error: {exc}")
            return False

    def trigger_summarization(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Trigger summarization for a session."""
        try:
            response = requests.post(
                f"{self.base_url}/api/chat/memory/summarize/{session_id}",
                timeout=60,  # Summarization can take time
            )
            if response.status_code == 200:
                return response.json()
            return None
        except requests.RequestException as exc:
            logger.error(f"Summarization error: {exc}")
            return None


# Singleton instance
chat_service = ChatService()
