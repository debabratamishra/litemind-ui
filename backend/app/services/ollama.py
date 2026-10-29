"""Backward-compatible Ollama wrapper built on the LiteLLM SDK."""
from __future__ import annotations

import logging

from .llm_gateway import get_ollama_api_base, stream_completion

logger = logging.getLogger(__name__)


def get_ollama_url():
    """Get the active Ollama API base URL."""
    url = get_ollama_api_base()
    logger.debug("Resolved Ollama URL: %s", url)
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
    """Stream markdown text from Ollama through LiteLLM."""
    async for chunk in stream_completion(
        messages,
        backend="ollama",
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        repetition_penalty=repetition_penalty,
    ):
        yield chunk
