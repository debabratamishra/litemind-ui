"""Utility helpers for realtime voice chat.

Includes PCM/WAV conversion, audio decoding, and Streamlit session-state
helpers used by the render orchestration.
"""

from __future__ import annotations

import io
import logging
from typing import Tuple

import numpy as np
import streamlit as st

from ...utils.memory_manager import ChatMemoryManager, RAGMemoryManager
from ..shared_ui import (
    get_backend_request_config,
    get_default_model_for_backend,
    get_generation_config_from_session,
    is_hosted_backend,
)
from .config import REALTIME_GREETING_CHAT, REALTIME_GREETING_RAG

logger = logging.getLogger(__name__)


def _pcm16_to_wav_bytes(pcm16: bytes, sample_rate: int, channels: int = 1) -> bytes:
    """Convert raw PCM16 bytes to WAV format."""
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16)
    return buf.getvalue()


def _get_realtime_greeting(page_key: str) -> str:
    """Get the appropriate greeting message based on page."""
    if page_key == "rag":
        return REALTIME_GREETING_RAG
    return REALTIME_GREETING_CHAT


def _get_memory_manager(page_key: str):
    """Get the appropriate memory manager based on page type.

    Args:
        page_key: Either "chat" or "rag"

    Returns:
        The appropriate memory manager instance (ChatMemoryManager or RAGMemoryManager)
    """
    if page_key == "rag":
        return RAGMemoryManager()
    return ChatMemoryManager()


def _get_chat_config_from_session(page_key: str = "chat") -> dict:
    """Extract chat configuration from session state.

    Args:
        page_key: The page key ("chat" or "rag") to determine appropriate model selection.
    """
    backend_provider = st.session_state.get("current_backend", "ollama")
    generation_config = get_generation_config_from_session(page_key)

    # Use appropriate model key based on page
    if page_key == "rag":
        if is_hosted_backend(backend_provider):
            model = st.session_state.get(
                f"selected_{backend_provider}_rag_model",
                get_default_model_for_backend(backend_provider),
            )
        else:
            model = st.session_state.get("selected_ollama_model", "default")
    else:
        if is_hosted_backend(backend_provider):
            model = st.session_state.get(
                f"selected_{backend_provider}_chat_model",
                get_default_model_for_backend(backend_provider),
            )
        else:
            model = st.session_state.get("selected_chat_model", "default")

    backend_request = get_backend_request_config(backend_provider)

    return {
        "backend": backend_provider,
        "model": model,
        **generation_config,
        "api_base": backend_request.get("api_base"),
        "api_key": backend_request.get("api_key"),
    }


def _get_messages_key(page_key: str) -> str:
    """Get the session state key for messages based on page."""
    if page_key == "rag":
        return "rag_messages"
    return "chat_messages"


def _decode_audio_bytes_to_pcm16_mono(audio_bytes: bytes, *, target_sample_rate: int) -> Tuple[bytes, int]:
    """Decode common audio formats (mp3/wav) into PCM16 mono at target_sample_rate."""
    try:
        import av
        from av.audio.resampler import AudioResampler
    except ImportError as e:
        raise RuntimeError("PyAV (av) is required to decode TTS audio") from e

    if not audio_bytes:
        return b"", target_sample_rate

    try:
        container = av.open(io.BytesIO(audio_bytes))
    except Exception as e:
        logger.warning(f"PyAV failed to open audio bytes: {e}")
        return b"", target_sample_rate

    audio_streams = [s for s in container.streams if s.type == "audio"]
    if not audio_streams:
        return b"", target_sample_rate

    in_stream = audio_streams[0]
    resampler = AudioResampler(
        format="s16",
        layout="mono",
        rate=target_sample_rate,
    )

    out = bytearray()
    try:
        for packet in container.demux(in_stream):
            for frame in packet.decode():
                for out_frame in resampler.resample(frame):
                    pcm = out_frame.to_ndarray()
                    if pcm.ndim == 2:
                        pcm = pcm[0]
                    if pcm.dtype != np.int16:
                        pcm = np.clip(pcm, -32768, 32767).astype(np.int16)
                    out.extend(pcm.tobytes())
    except Exception as e:
        logger.warning(f"PyAV decoding error: {e}")
        return b"", target_sample_rate

    return bytes(out), target_sample_rate
