"""Thin client to stream responses from a local Ollama server.

Notes
-----
* The function yields partial text chunks so callers can stream to clients.
* Retries with exponential backoff on transient network errors.
"""
from __future__ import annotations

import asyncio
import json
import re

import httpx

from config import Config


_OLLAMA_TIMEOUT = httpx.Timeout(connect=5.0, read=600.0, write=60.0, pool=None)
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0  # seconds


async def stream_ollama(messages, model: str = "gemma3n:e2b", temperature: float = 0.7):
    """Stream markdown text from Ollama.

    Parameters
    ----------
    messages : list[dict]
        Chat messages (role/content dicts) to send to the model.
    model : str
        Ollama model name (e.g. "llama3", "gemma3n:e2b").
    temperature : float
        Sampling temperature.

    Yields
    ------
    str
        Partial text chunks suitable for streaming to a client.
    """
    url = f"{Config.OLLAMA_API_URL}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": temperature},
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
