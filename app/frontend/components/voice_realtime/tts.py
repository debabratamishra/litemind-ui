"""Streaming TTS handler and greeting/playback helpers for realtime voice chat."""

from __future__ import annotations

import io
import logging
import queue
import threading
import time
from typing import Optional

import requests
import streamlit as st

from ...config import FASTAPI_URL
from .config import (
    MIN_TTS_CHUNK_SIZE,
    SENTENCE_ENDINGS,
    _greeting_audio_cache,
    _greeting_cache_lock,
)
from .utils import _decode_audio_bytes_to_pcm16_mono, _get_realtime_greeting

logger = logging.getLogger(__name__)


class StreamingTTSHandler:
    """
    Handles streaming TTS synthesis alongside streaming LLM responses.

    This class buffers incoming text and synthesizes audio sentence-by-sentence
    to reduce time-to-first-audio latency. Uses a background thread with an
    ordered queue to ensure audio chunks are synthesized and queued in order.
    Supports interruption for barge-in scenarios.
    """

    def __init__(self, audio_queue: queue.Queue, target_sample_rate: int = 48000):
        self._buffer = ""
        self._audio_queue = audio_queue  # Output queue for audio chunks
        self._target_sample_rate = target_sample_rate
        self._synthesized_sentences: list = []
        self._interrupted = threading.Event()  # Flag to stop synthesis on interruption
        self._full_text = ""  # Track complete text for saving

        # Ordered synthesis queue - sentences are synthesized in order
        self._sentence_queue: queue.Queue[Optional[str]] = queue.Queue()
        self._synthesis_thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()

        # Start the background synthesis thread
        self._start_synthesis_thread()

    def _start_synthesis_thread(self) -> None:
        """Start the background thread that synthesizes sentences in order."""

        def synthesis_worker():
            while not self._shutdown.is_set():
                try:
                    # Wait for next sentence with timeout
                    sentence = self._sentence_queue.get(timeout=0.1)

                    if sentence is None:  # Shutdown signal
                        break

                    if self._interrupted.is_set():
                        continue  # Skip if interrupted

                    # Synthesize this sentence (blocking, maintains order)
                    self._synthesize_sentence(sentence)

                except queue.Empty:
                    continue  # Keep waiting
                except Exception as e:
                    logger.debug(f"Synthesis worker error: {e}")

        self._synthesis_thread = threading.Thread(target=synthesis_worker, daemon=True)
        self._synthesis_thread.start()

    def _synthesize_sentence(self, text: str) -> None:
        """Synthesize a single sentence and add to audio queue (called from worker thread)."""
        if self._interrupted.is_set() or not text:
            return

        try:
            # Use the chunk synthesis endpoint for lower latency
            resp = requests.post(
                f"{FASTAPI_URL}/api/tts/synthesize-chunk",
                json={"text": text, "voice": None},
                timeout=30,
            )

            # Check again after network request
            if self._interrupted.is_set():
                return

            if resp.status_code == 200 and resp.content:
                pcm_out, _ = _decode_audio_bytes_to_pcm16_mono(
                    resp.content, target_sample_rate=self._target_sample_rate
                )
                if pcm_out and not self._interrupted.is_set():
                    self._audio_queue.put(pcm_out)
                    logger.debug(f"Queued TTS chunk: {len(pcm_out)} bytes for '{text[:30]}...'")
            else:
                logger.warning(f"TTS chunk synthesis failed: {resp.status_code}")
        except Exception as e:
            logger.error(f"TTS chunk error: {e}")

    def interrupt(self) -> None:
        """
        Signal that synthesis should stop (barge-in occurred).
        Clears pending work and prevents new synthesis.
        """
        self._interrupted.set()

        # Clear the sentence queue
        while True:
            try:
                self._sentence_queue.get_nowait()
            except queue.Empty:
                break

        # Clear the audio queue
        while True:
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        self._buffer = ""
        logger.debug("StreamingTTSHandler interrupted")

    def is_interrupted(self) -> bool:
        """Check if synthesis has been interrupted."""
        return self._interrupted.is_set()

    def get_full_text(self) -> str:
        """Get the complete text that was fed to the handler."""
        return self._full_text

    def feed(self, text_chunk: str) -> None:
        """
        Feed a text chunk from the LLM stream.

        Automatically queues complete sentences for synthesis in order.
        """
        if not text_chunk or self._interrupted.is_set():
            return

        self._buffer += text_chunk
        self._full_text += text_chunk  # Track for saving

        # Check for complete sentences
        sentences = SENTENCE_ENDINGS.split(self._buffer)

        if len(sentences) > 1:
            # Queue all complete sentences for synthesis (in order)
            for sentence in sentences[:-1]:
                if self._interrupted.is_set():
                    return
                sentence = sentence.strip()
                if sentence and len(sentence) >= MIN_TTS_CHUNK_SIZE:
                    self._sentence_queue.put(sentence)
                    self._synthesized_sentences.append(sentence)

            # Keep the incomplete sentence in buffer
            self._buffer = sentences[-1]

    def finalize(self) -> None:
        """Synthesize any remaining text in the buffer and wait for completion."""
        if self._interrupted.is_set():
            return

        if self._buffer.strip():
            # Queue final chunk
            self._sentence_queue.put(self._buffer.strip())
            self._synthesized_sentences.append(self._buffer.strip())
            self._buffer = ""

        # Wait for synthesis queue to drain (with timeout)
        timeout = 30.0  # Max wait time
        start_time = time.time()
        while not self._sentence_queue.empty() and (time.time() - start_time) < timeout:
            if self._interrupted.is_set():
                break
            time.sleep(0.1)

    def shutdown(self) -> None:
        """Shutdown the synthesis thread."""
        self._interrupted.set()
        self._shutdown.set()
        self._sentence_queue.put(None)  # Signal worker to exit

        if self._synthesis_thread and self._synthesis_thread.is_alive():
            self._synthesis_thread.join(timeout=2.0)


def _synthesize_and_play(text: str, voice: Optional[str] = None) -> None:
    """Synthesize text to speech and play it using st.audio."""
    if not text or not text.strip():
        return

    try:
        resp = requests.post(
            f"{FASTAPI_URL}/api/tts/synthesize",
            json={"text": text, "voice": voice, "use_cache": False},
            timeout=120,
        )
        if resp.status_code != 200:
            logger.warning("TTS synthesis failed: %s %s", resp.status_code, resp.text)
            st.error(f"TTS Failed: {resp.status_code} - {resp.text}")
            return

        audio_bytes = resp.content
        if not audio_bytes:
            st.warning("TTS returned no audio content")
            return

        content_type = (resp.headers.get("content-type") or "").lower()
        fmt = "audio/mp3"
        if "wav" in content_type:
            fmt = "audio/wav"
        elif "mpeg" in content_type or "mp3" in content_type:
            fmt = "audio/mp3"

        st.audio(io.BytesIO(audio_bytes), format=fmt, autoplay=True)
    except Exception as e:
        logger.debug("TTS playback error: %s", e)
        st.error(f"TTS Playback Error: {e}")


def _get_or_cache_greeting_audio(page_key: str = "chat") -> Optional[bytes]:
    """
    Get cached greeting audio or synthesize and cache it.

    This pre-caches the greeting audio to eliminate TTS latency on startup.
    Uses a thread-safe cache to avoid ScriptRunContext warnings from background threads.

    Args:
        page_key: The page key ("chat" or "rag") to determine appropriate greeting.
    """
    cache_key = f"greeting_{page_key}"
    greeting_text = _get_realtime_greeting(page_key)

    # Check thread-safe cache first
    with _greeting_cache_lock:
        if cache_key in _greeting_audio_cache:
            cached = _greeting_audio_cache[cache_key]
            if cached:
                logger.debug(f"Using cached greeting audio for {page_key}")
                return cached

    # Synthesize greeting audio
    try:
        logger.info(f"Synthesizing greeting audio for {page_key}...")
        resp = requests.post(
            f"{FASTAPI_URL}/api/tts/synthesize",
            json={"text": greeting_text, "voice": None, "use_cache": True},
            timeout=30,
        )
        if resp.status_code == 200 and resp.content:
            # Store in thread-safe cache
            with _greeting_cache_lock:
                _greeting_audio_cache[cache_key] = resp.content
            logger.info(f"Greeting audio cached: {len(resp.content)} bytes")
            return resp.content
        else:
            logger.warning(f"Failed to synthesize greeting: {resp.status_code}")
    except Exception as e:
        logger.error(f"Error caching greeting audio: {e}")

    return None


def _play_greeting_via_webrtc(audio_processor, audio_bytes: bytes, target_sr: int = 48000) -> bool:
    """
    Play greeting audio through WebRTC audio processor.

    Returns True if successful.
    """
    if not audio_bytes or not hasattr(audio_processor, "enqueue_assistant_pcm16"):
        return False

    try:
        pcm_out, _ = _decode_audio_bytes_to_pcm16_mono(audio_bytes, target_sample_rate=target_sr)
        if pcm_out:
            audio_processor.enqueue_assistant_pcm16(pcm_out)
            logger.info("Greeting audio enqueued for playback")
            return True
    except Exception as e:
        logger.error(f"Failed to play greeting via WebRTC: {e}")

    return False


def _play_greeting_via_audio_widget(audio_bytes: bytes) -> bool:
    """
    Play greeting audio using st.audio widget as fallback.

    Returns True if successful.
    """
    if not audio_bytes:
        return False

    try:
        # Detect format from content
        fmt = "audio/mp3"
        if audio_bytes[:4] == b"RIFF":
            fmt = "audio/wav"

        st.audio(io.BytesIO(audio_bytes), format=fmt, autoplay=True)
        logger.info("Greeting played via audio widget")
        return True
    except Exception as e:
        logger.error(f"Failed to play greeting via audio widget: {e}")

    return False
