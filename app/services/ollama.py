"""Thin client to stream responses from a local Ollama server.

Notes
-----
* The function yields partial text chunks so callers can stream to clients.
* Retries with exponential backoff on transient network errors.
* Supports dynamic configuration for container vs native execution.
"""
from __future__ import annotations

import asyncio
import json
import re
import logging

import httpx

from config import Config

logger = logging.getLogger(__name__)


_OLLAMA_TIMEOUT = httpx.Timeout(connect=5.0, read=600.0, write=60.0, pool=None)
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0  # seconds


def get_ollama_url():
    """Get the appropriate Ollama URL based on execution environment."""
    try:
        from app.services.host_service_manager import host_service_manager
        url = host_service_manager.environment_config.ollama_url
        logger.debug(f"Using Ollama URL from host service manager: {url}")
        return url
    except ImportError:
        logger.warning("Host service manager not available, using fallback config")
        # Fallback to config-based detection
        from config import Config
        url = Config.OLLAMA_API_URL
        logger.debug(f"Using fallback Ollama URL: {url}")
        return url


async def stream_ollama(
    messages, 
    model: str = "gemma3:1b", 
    temperature: float = 0.7, 
    max_tokens: int = 2048,
    top_p: float = 0.9,
    frequency_penalty: float = 0.0,
    repetition_penalty: float = 1.0
):
    """Stream markdown text from Ollama.

    Parameters
    ----------
    messages : list[dict]
        Chat messages (role/content dicts) to send to the model.
    model : str
        Ollama model name (e.g. "llama3", "gemma3:1b").
    temperature : float
        Sampling temperature.
    max_tokens : int
        Maximum number of tokens to generate.
    top_p : float
        Nucleus sampling parameter (0.0 to 1.0).
    frequency_penalty : float
        Penalize frequent tokens (-2.0 to 2.0). Maps to Ollama's frequency_penalty.
    repetition_penalty : float
        Penalize repeated tokens (0.0 to 2.0). Maps to Ollama's repeat_penalty.

    Yields
    ------
    str
        Partial text chunks suitable for streaming to a client.
    """
    ollama_url = get_ollama_url()
    url = f"{ollama_url}/api/chat"
    logger.info(f"Using Ollama URL: {ollama_url}")
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": temperature, 
            "num_predict": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "repeat_penalty": repetition_penalty,
        },
    }

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=_OLLAMA_TIMEOUT) as client:
                async with client.stream("POST", url, json=payload) as resp:
                    resp.raise_for_status()
                    buffer = ""
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        content = data.get("message", {}).get("content", "")
                        if not content:
                            continue
                        buffer += content
                        if re.search(r"[.!?\n]$", buffer) or data.get("done", False) or len(buffer) > 400:
                            yield buffer
                            buffer = ""
                    if buffer:
                        yield buffer
            return  # success
        except (httpx.ReadTimeout, httpx.ConnectTimeout):
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_RETRY_BACKOFF * attempt)
                continue
            yield "*The request to Ollama **timed out**. Please retry or simplify the prompt.*"
            return
        except (httpx.RequestError, httpx.HTTPStatusError):
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_RETRY_BACKOFF * attempt)
                continue
            yield "*Could not reach Ollama at `http://localhost:11434`. Is it running?*"
            return
