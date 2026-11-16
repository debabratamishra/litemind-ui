"""Text-to-Speech service using local voices.

This module provides offline TTS via pyttsx3 (system voices). No external
model download or network call is required.

Features:
- Streaming TTS: Generate audio chunks as text streams in
- Model preloading: No-op kept for startup compatibility
- Text preprocessing: Clean text for natural speech output
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import tempfile
import unicodedata
from typing import Any, AsyncGenerator, Optional, Tuple

from ..core.text_markup import remove_tagged_sections, replace_fenced_code_blocks

# Import text processing helpers
from .tts_text_processing import (
    _compress_repeated_chars,
    _is_emoji_codepoint,
    _normalize_whitespace,
    _remove_brace_blocks,
    _remove_file_paths,
    _remove_inline_code,
    _remove_list_markers,
    _remove_markdown_formatting,
    _remove_markdown_headers,
    _remove_markdown_links,
    _remove_special_chars,
    _remove_urls,
    _split_on_sentence_endings,
    _strip_html_tags,
)

logger = logging.getLogger(__name__)

# TTS Configuration
TTS_CONFIG: dict[str, Any] = {
    "default_voice": "default",
}

# Cache directory for TTS audio
CACHE_DIR = os.path.join(tempfile.gettempdir(), "litemind_tts_cache")

# Minimum characters before attempting to synthesize a chunk
# Lower values = faster first audio, but more synthesis calls
MIN_CHUNK_SIZE = 20
MAX_CHUNK_SIZE = 300


class TTSService:
    """Text-to-Speech service backed by pyttsx3 local voices."""

    def __init__(self, preload_models: bool = False):
        self._pyttsx3_available = False
        self._pyttsx3_engine = None
        self._check_backends()
        self._ensure_cache_dir()

    def _ensure_cache_dir(self):
        """Ensure cache directory exists."""
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create TTS cache directory: {e}")

    def _check_backends(self):
        """Check whether the pyttsx3 backend is available."""
        try:
            import pyttsx3  # noqa: F401
            self._pyttsx3_available = True
            logger.info("pyttsx3 TTS backend available")
        except ImportError:
            logger.warning("pyttsx3 not installed")

    def preload(self):
        """Compatibility no-op — pyttsx3 needs no model preload."""
        return None

    def is_model_loaded(self) -> bool:
        """pyttsx3 loads on demand, so there is no resident model."""
        return False

    def _clean_text_for_tts(self, text: str) -> str:
        """
        Clean text for TTS by removing markdown, emojis, code, and special formatting.

        This method handles:
        - Thinking/reasoning tags
        - Code blocks and inline code
        - Markdown formatting (bold, italic, links, headers, lists)
        - HTML tags
        - Emojis and special unicode characters
        - URLs
        - File paths
        - Technical artifacts (JSON, XML, etc.)

        All operations are O(n) single-pass string operations — no regex,
        no backtracking, immune to ReDoS.
        """
        if not text:
            return ""

        # Remove thinking/reasoning tags
        text = remove_tagged_sections(text, ("think", "thinking", "reasoning", "thought"))

        # Remove markdown code blocks - replace with spoken indication
        text = replace_fenced_code_blocks(text, " [code block omitted] ")

        # Remove inline code
        text = _remove_inline_code(text)

        # Remove URLs
        text = _remove_urls(text)

        # Remove file paths (Unix and Windows style)
        text = _remove_file_paths(text)

        # Remove JSON-like structures
        text = _remove_brace_blocks(text, '{', '}')
        text = _remove_brace_blocks(text, '[', ']')

        # Remove markdown formatting (bold, italic, strikethrough)
        text = _remove_markdown_formatting(text)

        # Remove markdown links but keep text
        text = _remove_markdown_links(text)

        # Remove headers
        text = _remove_markdown_headers(text)

        # Remove bullet points and list markers
        text = _remove_list_markers(text)

        # Remove HTML tags (no regex — avoids ReDoS on malformed input)
        text = _strip_html_tags(text)

        # Remove emojis and other non-speech characters
        text = self._remove_emojis(text)

        # Remove special characters that don't sound good when read
        text = _remove_special_chars(text)

        # Clean up quotes - keep the content but remove excessive quoting
        text = _compress_repeated_chars(text, frozenset({'"', "'"}), '"')

        # Replace multiple dashes/underscores with space
        text = _compress_repeated_chars(text, frozenset({'-', '_'}), ' ')

        # Clean up extra whitespace
        text = _normalize_whitespace(text)

        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(line for line in lines if line)

        return text.strip()

    def _remove_emojis(self, text: str) -> str:
        """Remove emojis and other non-speech characters from text.

        Uses codepoint range checks (O(1) per character) instead of regex
        for emoji detection, followed by unicodedata category filtering.
        """
        cleaned = []
        for char in text:
            cp = ord(char)

            # Skip known emoji codepoints
            if _is_emoji_codepoint(cp):
                continue

            category = unicodedata.category(char)
            # Keep letters, numbers, punctuation, and basic whitespace
            if category.startswith(('L', 'N', 'P', 'Z')) or char in ' \n\t':
                cleaned.append(char)
            elif category == 'So':  # Other symbols - skip most
                continue
            else:
                cleaned.append(char)

        return ''.join(cleaned)

    def _split_into_sentences(self, text: str) -> list:
        """Split text into sentences for streaming synthesis."""
        if not text:
            return []

        # Split on sentence endings while keeping the punctuation
        sentences = _split_on_sentence_endings(text)

        # Filter out empty strings and very short fragments
        result = []
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            current += " " + sentence if current else sentence

            # If we have enough text, yield it
            if len(current) >= MIN_CHUNK_SIZE:
                result.append(current.strip())
                current = ""

        # Don't forget the last chunk
        if current.strip():
            result.append(current.strip())

        return result

    def _get_cache_key(self, text: str, voice: str) -> str:
        """Generate a cache key for the text and voice combination."""
        content = f"{text}:{voice}".encode('utf-8')
        return hashlib.md5(content).hexdigest()

    def _get_cached_audio(self, cache_key: str) -> Optional[bytes]:
        """Get cached audio if available."""
        cache_path = os.path.join(CACHE_DIR, f"{cache_key}.mp3")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return f.read()
            except Exception:
                pass
        return None

    def _cache_audio(self, cache_key: str, audio_data: bytes):
        """Cache audio data."""
        cache_path = os.path.join(CACHE_DIR, f"{cache_key}.mp3")
        try:
            with open(cache_path, 'wb') as f:
                f.write(audio_data)
        except Exception as e:
            logger.warning(f"Failed to cache audio: {e}")

    def _synthesize_pyttsx3(self, text: str) -> Optional[bytes]:
        """Synthesize speech using pyttsx3 (offline, system voices)."""
        try:
            import pyttsx3

            engine = pyttsx3.init()

            # Create temporary file for output
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name

            try:
                engine.save_to_file(text, tmp_path)
                engine.runAndWait()

                # Read the file
                with open(tmp_path, 'rb') as f:
                    audio_data = f.read()

                return audio_data
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"pyttsx3 synthesis failed: {e}")
            return None

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        use_cache: bool = True
    ) -> Tuple[Optional[bytes], str]:
        """
        Synthesize speech from text.

        Args:
            text: Text to convert to speech
            voice: Voice ID (unused; kept for API compatibility)
            use_cache: Whether to use caching (currently unused)

        Returns:
            Tuple of (audio_bytes, content_type)
        """
        logger.info(f"TTS synthesize called: text_length={len(text) if text else 0}, voice={voice}")

        if not text or not text.strip():
            logger.warning("TTS: Empty text provided")
            return None, ""

        # Clean text for TTS
        clean_text = self._clean_text_for_tts(text)
        if not clean_text:
            logger.warning("TTS: Text was empty after cleaning")
            return None, ""

        logger.info(f"TTS: Cleaned text length={len(clean_text)}")

        # Limit text length to avoid very long audio
        if len(clean_text) > 5000:
            clean_text = clean_text[:5000] + "... Text truncated for speech output."

        if self._pyttsx3_available:
            logger.info("TTS: Using pyttsx3 backend")
            loop = asyncio.get_running_loop()
            audio_data = await loop.run_in_executor(None, self._synthesize_pyttsx3, clean_text)
            if audio_data:
                logger.info(f"TTS: pyttsx3 success, {len(audio_data)} bytes")
                return audio_data, "audio/wav"
            logger.warning("TTS: pyttsx3 returned no data")

        logger.error("TTS: No backend available or synthesis failed")
        return None, ""

    async def synthesize_streaming(
        self,
        text_generator: AsyncGenerator[str, None],
        voice: Optional[str] = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Synthesize speech from streaming text input.

        Yields audio chunks as sentences complete.

        Args:
            text_generator: Async generator yielding text chunks
            voice: Voice ID (unused; kept for API compatibility)
        """
        buffer = ""

        async for text_chunk in text_generator:
            if not text_chunk:
                continue

            # Clean the incoming chunk
            clean_chunk = self._clean_text_for_tts(text_chunk)
            if not clean_chunk:
                continue

            buffer += clean_chunk

            # Check if we have a complete sentence or enough text
            sentences = self._split_into_sentences(buffer)

            if len(sentences) > 1:
                # Synthesize all complete sentences
                for sentence in sentences[:-1]:
                    if self._pyttsx3_available:
                        loop = asyncio.get_running_loop()
                        audio = await loop.run_in_executor(
                            None, lambda s=sentence: self._synthesize_pyttsx3(s)
                        )
                        if audio:
                            yield audio

                # Keep the last incomplete sentence in buffer
                buffer = sentences[-1]
            elif len(buffer) > MAX_CHUNK_SIZE:
                # Force synthesis if buffer is too large
                if self._pyttsx3_available:
                    loop = asyncio.get_running_loop()
                    audio = await loop.run_in_executor(None, self._synthesize_pyttsx3, buffer)
                    if audio:
                        yield audio
                buffer = ""

        # Synthesize any remaining text
        if buffer.strip() and self._pyttsx3_available:
            loop = asyncio.get_running_loop()
            audio = await loop.run_in_executor(None, self._synthesize_pyttsx3, buffer)
            if audio:
                yield audio

    def synthesize_text_chunk(self, text: str, voice: Optional[str] = None) -> Optional[bytes]:
        """
        Synchronously synthesize a single text chunk.

        Args:
            text: Text chunk to synthesize
            voice: Voice ID (unused; kept for API compatibility)

        Returns:
            Audio bytes or None
        """
        if not text or not text.strip():
            return None

        clean_text = self._clean_text_for_tts(text)
        if not clean_text:
            return None

        if self._pyttsx3_available:
            return self._synthesize_pyttsx3(clean_text)

        return None

    def synthesize_sync(
        self,
        text: str,
        voice: Optional[str] = None,
        use_cache: bool = True
    ) -> Tuple[Optional[bytes], str]:
        """Synchronous wrapper for synthesize."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.synthesize(text, voice, use_cache))

    def get_available_voices(self) -> list:
        """Get list of available local voices (pyttsx3 is system-dependent)."""
        return [{"id": "default", "name": "System default"}]

    def is_available(self) -> bool:
        """Check if the pyttsx3 backend is available."""
        return self._pyttsx3_available

    def get_status(self) -> dict:
        """Get TTS service status."""
        return {
            "available": self.is_available(),
            "backend": "pyttsx3",
            "pyttsx3": self._pyttsx3_available,
            "default_voice": TTS_CONFIG["default_voice"],
        }


# Global singleton instance
_tts_service: Optional[TTSService] = None


def get_tts_service(preload: bool = False) -> TTSService:
    """
    Get or create the global TTS service instance.

    Args:
        preload: Kept for API compatibility; pyttsx3 needs no preload.
    """
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService(preload_models=preload)
    return _tts_service


def preload_tts_model():
    """Compatibility no-op — pyttsx3 needs no model preload."""
    logger.info("TTS model preloaded")
