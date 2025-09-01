"""
Chat service for handling chat interactions and streaming.
"""
import asyncio
import logging
import requests
from typing import Dict, List, Optional, Any

from ..config import FASTAPI_URL, CONNECT_TIMEOUT, READ_TIMEOUT
from ...services.ollama import stream_ollama

logger = logging.getLogger(__name__)


class ChatService:
    """Service for handling chat interactions."""
    
    def __init__(self):
        self.base_url = FASTAPI_URL

    def call_fastapi_chat(
        self, 
        message: str, 
        model: str = "default", 
        temperature: float = 0.7
    ) -> Optional[str]:
        """Call the non-streaming chat endpoint."""
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={"message": message, "model": model, "temperature": temperature},
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
        backend: str = "ollama",
        hf_token: Optional[str] = None
    ) -> requests.Response:
        """Stream a chat response from the backend."""
        payload = {
            "message": message,
            "model": model,
            "temperature": temperature,
            "backend": backend
        }
        
        if backend == "vllm" and hf_token:
            payload["hf_token"] = hf_token
        
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
        temperature: float
    ) -> str:
        """Stream a chat response from local Ollama."""
        async def _inner() -> str:
            acc = ""
            async for chunk in stream_ollama(
                [{"role": "user", "content": message}], 
                model=model, 
                temperature=temperature
            ):
                acc += chunk
            return acc

        return asyncio.run(_inner())


# Singleton instance
chat_service = ChatService()
