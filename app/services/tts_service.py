"""
Text-to-Speech service using lightweight open-source models.

This module provides TTS functionality using:
1. Kokoro TTS (High quality, offline, fast) - Preferred
2. Edge TTS (Microsoft Edge's TTS - fast, high quality, no API key needed)
3. pyttsx3 as fallback (offline, system voices)

Kokoro is preferred as it's offline, high quality, and supports streaming.

Features:
- Streaming TTS: Generate audio chunks as text streams in
- Model preloading: Load models at startup for reduced latency
- Text preprocessing: Clean text for natural speech output
"""

import asyncio
import io
import logging
import os
import re
import tempfile
from typing import Optional, Tuple, AsyncGenerator, Generator
import hashlib
import unicodedata

logger = logging.getLogger(__name__)

# TTS Configuration
TTS_CONFIG = {
    "default_voice": "en-US-AriaNeural",  # Natural female voice (Edge TTS)
    "kokoro_voice": "af_heart",           # Default Kokoro voice
    "alternative_voices": [
        "en-US-GuyNeural",      # Male voice
        "en-GB-SoniaNeural",    # British female
        "en-AU-NatashaNeural",  # Australian female
    ],
    "rate": "+0%",  # Speech rate adjustment
    "volume": "+0%",  # Volume adjustment
    "pitch": "+0Hz",  # Pitch adjustment
}

# Cache directory for TTS audio
CACHE_DIR = os.path.join(tempfile.gettempdir(), "litemind_tts_cache")

# Sentence-ending punctuation for streaming chunking
SENTENCE_ENDINGS = re.compile(r'[.!?;:]\s*')

# Minimum characters before attempting to synthesize a chunk
MIN_CHUNK_SIZE = 50
MAX_CHUNK_SIZE = 500


class TTSService:
    """Text-to-Speech service with multiple backend support and streaming capabilities."""
    
    def __init__(self, preload_models: bool = False):
        """
        Initialize TTS service.
        
        Args:
            preload_models: If True, load Kokoro model immediately during init
        """
        self._preferred_backend = os.getenv("TTS_BACKEND", "auto").strip().lower()
        self._edge_tts_available = False
        self._pyttsx3_available = False
        self._kokoro_available = False
        self._edge_tts = None
        self._pyttsx3_engine = None
        self._kokoro_pipeline = None
        self._kokoro_loaded = False
        self._check_backends()
        self._ensure_cache_dir()
        
        if preload_models:
            self._preload_kokoro()
    
    def _ensure_cache_dir(self):
        """Ensure cache directory exists."""
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create TTS cache directory: {e}")
    
    def _check_backends(self):
        """Check which TTS backends are available."""
        # Check for Kokoro
        try:
            from kokoro import KPipeline
            import soundfile as sf
            self._kokoro_available = True
            logger.info("Kokoro TTS backend available")
        except ImportError:
            logger.warning("kokoro or soundfile not installed")

        # Check for edge-tts
        try:
            import edge_tts
            self._edge_tts_available = True
            self._edge_tts = edge_tts
            logger.info("Edge TTS backend available")
        except ImportError:
            logger.warning("edge-tts not installed, will try fallback")
        
        # Check for pyttsx3 fallback
        try:
            import pyttsx3
            self._pyttsx3_available = True
            logger.info("pyttsx3 fallback available")
        except ImportError:
            logger.warning("pyttsx3 not installed")
    
    def _preload_kokoro(self):
        """Preload Kokoro model into memory."""
        if not self._kokoro_available or self._kokoro_loaded:
            return
        
        try:
            from kokoro import KPipeline
            logger.info("Preloading Kokoro TTS model...")
            self._kokoro_pipeline = KPipeline(lang_code='a')
            self._kokoro_loaded = True
            logger.info("Kokoro TTS model preloaded successfully")
        except Exception as e:
            logger.error(f"Failed to preload Kokoro model: {e}")
    
    def preload(self):
        """Public method to preload models. Call during app startup."""
        self._preload_kokoro()
    
    def is_model_loaded(self) -> bool:
        """Check if the TTS model is loaded in memory."""
        return self._kokoro_loaded
    
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
        """
        if not text:
            return ""
        
        # Remove thinking/reasoning tags
        text = re.sub(r'<\s*(think|thinking|reasoning|thought)\s*>.*?<\s*/\s*(think|thinking|reasoning|thought)\s*>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove markdown code blocks - replace with spoken indication
        text = re.sub(r'```[\w]*\n[\s\S]*?```', ' [code block omitted] ', text)
        text = re.sub(r'```[\s\S]*?```', ' [code block omitted] ', text)
        
        # Remove inline code
        text = re.sub(r'`[^`]+`', '', text)
        
        # Remove URLs
        text = re.sub(r'https?://[^\s<>"{}|\\^`\[\]]+', ' [link] ', text)
        text = re.sub(r'www\.[^\s<>"{}|\\^`\[\]]+', ' [link] ', text)
        
        # Remove file paths (Unix and Windows style)
        text = re.sub(r'(?:/[\w.-]+)+/?', '', text)
        text = re.sub(r'(?:[A-Za-z]:\\[\w\\.-]+)+', '', text)
        
        # Remove JSON-like structures
        text = re.sub(r'\{[^{}]*\}', '', text)
        text = re.sub(r'\[[^\[\]]*\]', '', text)
        
        # Remove markdown formatting
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Italic
        text = re.sub(r'__([^_]+)__', r'\1', text)  # Bold
        text = re.sub(r'_([^_]+)_', r'\1', text)  # Italic
        text = re.sub(r'~~([^~]+)~~', r'\1', text)  # Strikethrough
        
        # Remove markdown links but keep text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # Remove headers
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        
        # Remove bullet points and list markers
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove emojis and special unicode characters
        text = self._remove_emojis(text)
        
        # Remove special characters that don't sound good when read
        text = re.sub(r'[#@$%^&*()+=\[\]{}|\\<>~]', ' ', text)
        
        # Clean up quotes - keep the content but remove excessive quoting
        text = re.sub(r'["\']{2,}', '"', text)
        
        # Replace multiple dashes/underscores with space
        text = re.sub(r'[-_]{2,}', ' ', text)
        
        # Clean up extra whitespace
        text = re.sub(r'\n\s*\n', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(line for line in lines if line)
        
        return text.strip()
    
    def _remove_emojis(self, text: str) -> str:
        """Remove emojis and other non-speech characters from text."""
        # Remove emoji characters
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"  # dingbats
            "\U000024C2-\U0001F251"  # enclosed characters
            "\U0001F900-\U0001F9FF"  # supplemental symbols
            "\U0001FA00-\U0001FA6F"  # chess symbols
            "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
            "\U00002600-\U000026FF"  # misc symbols
            "\U00002700-\U000027BF"  # dingbats
            "\U0001F000-\U0001F02F"  # mahjong tiles
            "\U0001F0A0-\U0001F0FF"  # playing cards
            "]+",
            flags=re.UNICODE
        )
        text = emoji_pattern.sub('', text)
        
        # Remove other special unicode categories that aren't speech-friendly
        cleaned = []
        for char in text:
            category = unicodedata.category(char)
            # Keep letters, numbers, punctuation, and basic symbols
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
        sentences = SENTENCE_ENDINGS.split(text)
        
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

    def _synthesize_kokoro(self, text: str, voice: str = None) -> Optional[bytes]:
        """Synthesize speech using Kokoro TTS."""
        try:
            from kokoro import KPipeline
            import soundfile as sf
            import torch
            
            if self._kokoro_pipeline is None:
                logger.info("Initializing Kokoro pipeline (first run may take a moment)...")
                self._kokoro_pipeline = KPipeline(lang_code='a')
                self._kokoro_loaded = True
            
            voice = voice or TTS_CONFIG["kokoro_voice"]
            
            # Kokoro generator yields (graphemes, phonemes, audio_tensor)
            generator = self._kokoro_pipeline(
                text, 
                voice=voice,
                speed=1, 
                split_pattern=r'\n+'
            )
            
            all_audio = []
            for i, (gs, ps, audio) in enumerate(generator):
                all_audio.append(audio)
            
            if not all_audio:
                return None
                
            # Concatenate all audio chunks
            final_audio = torch.cat(all_audio, dim=0)
            
            # Convert to WAV bytes
            buf = io.BytesIO()
            sf.write(buf, final_audio, 24000, format='WAV')
            return buf.getvalue()
            
        except Exception as e:
            logger.error(f"Kokoro synthesis failed: {e}")
            return None
    
    def _synthesize_kokoro_streaming(self, text: str, voice: str = None) -> Generator[bytes, None, None]:
        """
        Synthesize speech using Kokoro TTS with streaming output.
        
        Yields audio chunks as they are generated.
        """
        try:
            from kokoro import KPipeline
            import soundfile as sf
            import torch
            
            if self._kokoro_pipeline is None:
                logger.info("Initializing Kokoro pipeline for streaming...")
                self._kokoro_pipeline = KPipeline(lang_code='a')
                self._kokoro_loaded = True
            
            voice = voice or TTS_CONFIG["kokoro_voice"]
            
            # Kokoro generator yields (graphemes, phonemes, audio_tensor)
            generator = self._kokoro_pipeline(
                text, 
                voice=voice,
                speed=1, 
                split_pattern=r'[.!?;:]\s*'  # Split on sentence boundaries for streaming
            )
            
            for gs, ps, audio in generator:
                if audio is not None and len(audio) > 0:
                    # Convert each chunk to WAV bytes
                    buf = io.BytesIO()
                    sf.write(buf, audio, 24000, format='WAV')
                    yield buf.getvalue()
            
        except Exception as e:
            logger.error(f"Kokoro streaming synthesis failed: {e}")
    
    async def _synthesize_edge_tts(self, text: str, voice: str) -> Optional[bytes]:
        """Synthesize speech using Edge TTS."""
        try:
            logger.info(f"Starting Edge TTS synthesis: voice={voice}, text_length={len(text)}")
            communicate = self._edge_tts.Communicate(
                text,
                voice,
                rate=TTS_CONFIG["rate"],
                volume=TTS_CONFIG["volume"],
                pitch=TTS_CONFIG["pitch"]
            )
            
            audio_data = io.BytesIO()
            chunk_count = 0
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data.write(chunk["data"])
                    chunk_count += 1
            
            result = audio_data.getvalue()
            if not result:
                raise Exception("No audio data received from Edge TTS")

            logger.info(f"Edge TTS synthesis complete: {len(result)} bytes, {chunk_count} chunks")
            return result
        except Exception as e:
            logger.error(f"Edge TTS synthesis failed: {type(e).__name__}: {e}")
            return None
    
    async def _synthesize_edge_tts_streaming(self, text: str, voice: str) -> AsyncGenerator[bytes, None]:
        """
        Synthesize speech using Edge TTS with streaming output.
        
        Yields audio chunks as they are generated.
        """
        try:
            communicate = self._edge_tts.Communicate(
                text,
                voice,
                rate=TTS_CONFIG["rate"],
                volume=TTS_CONFIG["volume"],
                pitch=TTS_CONFIG["pitch"]
            )
            
            async for chunk in communicate.stream():
                if chunk["type"] == "audio" and chunk["data"]:
                    yield chunk["data"]
                    
        except Exception as e:
            logger.error(f"Edge TTS streaming failed: {e}")
    
    def _synthesize_pyttsx3(self, text: str) -> Optional[bytes]:
        """Synthesize speech using pyttsx3 (offline fallback)."""
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
                except:
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
            voice: Voice ID to use (optional, uses default if not specified)
            use_cache: Whether to use caching
            
        Returns:
            Tuple of (audio_bytes, content_type)
        """
        logger.info(f"TTS synthesize called: text_length={len(text) if text else 0}, voice={voice}, use_cache={use_cache}")
        
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
        
        voice = voice or TTS_CONFIG["default_voice"]

        preferred = self._preferred_backend
        prefer_kokoro = preferred in {"auto", "kokoro", "offline"}
        prefer_edge = preferred in {"auto", "edge", "edge-tts", "edgetts"}
        prefer_pyttsx3 = preferred in {"auto", "pyttsx3", "offline"}
        
        cache_key: Optional[str] = None

        # Try Kokoro first (High quality, offline)
        if prefer_kokoro and self._kokoro_available:
            logger.info("TTS: Using Kokoro backend")
            # Run in thread pool to avoid blocking event loop
            loop = asyncio.get_running_loop()
            audio_data = await loop.run_in_executor(None, self._synthesize_kokoro, clean_text)
            if audio_data:
                logger.info(f"TTS: Kokoro success, {len(audio_data)} bytes")
                return audio_data, "audio/wav"
            logger.error("TTS: Kokoro returned no data")

        # Cache only applies to MP3 (Edge TTS) in current implementation.
        if use_cache and prefer_edge and self._edge_tts_available:
            cache_key = self._get_cache_key(clean_text, voice)
            cached = self._get_cached_audio(cache_key)
            if cached:
                logger.info(f"TTS: Returning cached audio ({len(cached)} bytes)")
                return cached, "audio/mpeg"
        
        # Try Edge TTS (best quality online)
        if prefer_edge and self._edge_tts_available:
            logger.info("TTS: Using Edge TTS backend")
            audio_data = await self._synthesize_edge_tts(clean_text, voice)
            if audio_data:
                logger.info(f"TTS: Edge TTS success, {len(audio_data)} bytes")
                if use_cache and cache_key:
                    self._cache_audio(cache_key, audio_data)
                return audio_data, "audio/mpeg"
            logger.error("TTS: Edge TTS returned no data")
        
        # Fallback to pyttsx3
        if prefer_pyttsx3 and self._pyttsx3_available:
            logger.info("TTS: Falling back to pyttsx3")
            audio_data = self._synthesize_pyttsx3(clean_text)
            if audio_data:
                logger.info(f"TTS: pyttsx3 success, {len(audio_data)} bytes")
                return audio_data, "audio/wav"
            else:
                logger.error("TTS: pyttsx3 returned no data")
        
        logger.error("TTS: No backend available or all backends failed")
        return None, ""
    
    async def synthesize_streaming(
        self,
        text_generator: AsyncGenerator[str, None],
        voice: Optional[str] = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Synthesize speech from streaming text input.
        
        This method accepts an async generator of text chunks and yields
        audio chunks as they become available, significantly reducing
        time-to-first-audio.
        
        Args:
            text_generator: Async generator yielding text chunks
            voice: Voice ID to use
            
        Yields:
            Audio bytes chunks (WAV format for Kokoro, MP3 for Edge TTS)
        """
        # Use appropriate voice for each backend
        kokoro_voice = voice if voice and not voice.startswith("en-") else TTS_CONFIG["kokoro_voice"]
        edge_voice = voice or TTS_CONFIG["default_voice"]
        buffer = ""
        
        preferred = self._preferred_backend
        use_kokoro = preferred in {"auto", "kokoro", "offline"} and self._kokoro_available
        use_edge = preferred in {"auto", "edge", "edge-tts", "edgetts"} and self._edge_tts_available
        
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
                    if use_kokoro:
                        loop = asyncio.get_running_loop()
                        for audio_chunk in await loop.run_in_executor(
                            None, 
                            lambda s=sentence: list(self._synthesize_kokoro_streaming(s, kokoro_voice))
                        ):
                            yield audio_chunk
                    elif use_edge:
                        async for audio_chunk in self._synthesize_edge_tts_streaming(sentence, edge_voice):
                            yield audio_chunk
                
                # Keep the last incomplete sentence in buffer
                buffer = sentences[-1]
            elif len(buffer) > MAX_CHUNK_SIZE:
                # Force synthesis if buffer is too large
                if use_kokoro:
                    loop = asyncio.get_running_loop()
                    for audio_chunk in await loop.run_in_executor(
                        None,
                        lambda: list(self._synthesize_kokoro_streaming(buffer, kokoro_voice))
                    ):
                        yield audio_chunk
                elif use_edge:
                    async for audio_chunk in self._synthesize_edge_tts_streaming(buffer, edge_voice):
                        yield audio_chunk
                buffer = ""
        
        # Synthesize any remaining text
        if buffer.strip():
            if use_kokoro:
                loop = asyncio.get_running_loop()
                for audio_chunk in await loop.run_in_executor(
                    None,
                    lambda: list(self._synthesize_kokoro_streaming(buffer, kokoro_voice))
                ):
                    yield audio_chunk
            elif use_edge:
                async for audio_chunk in self._synthesize_edge_tts_streaming(buffer, edge_voice):
                    yield audio_chunk
    
    def synthesize_text_chunk(self, text: str, voice: Optional[str] = None) -> Optional[bytes]:
        """
        Synchronously synthesize a single text chunk.
        
        Useful for streaming scenarios where you want to synthesize
        sentence by sentence.
        
        Args:
            text: Text chunk to synthesize
            voice: Voice ID to use
            
        Returns:
            Audio bytes or None
        """
        if not text or not text.strip():
            return None
        
        clean_text = self._clean_text_for_tts(text)
        if not clean_text:
            return None
        
        preferred = self._preferred_backend
        
        if preferred in {"auto", "kokoro", "offline"} and self._kokoro_available:
            return self._synthesize_kokoro(clean_text, voice)
        
        # For Edge TTS, we need to run async
        if preferred in {"auto", "edge", "edge-tts"} and self._edge_tts_available:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return loop.run_until_complete(self._synthesize_edge_tts(clean_text, voice or TTS_CONFIG["default_voice"]))
        
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
        """Get list of available voices."""
        voices = [TTS_CONFIG["default_voice"]] + TTS_CONFIG["alternative_voices"]
        return [{"id": v, "name": v.replace("-", " ").replace("Neural", "")} for v in voices]
    
    def is_available(self) -> bool:
        """Check if any TTS backend is available."""
        return self._kokoro_available or self._edge_tts_available or self._pyttsx3_available
    
    def get_status(self) -> dict:
        """Get TTS service status."""
        return {
            "available": self.is_available(),
            "preferred_backend": self._preferred_backend,
            "kokoro": self._kokoro_available,
            "kokoro_loaded": self._kokoro_loaded,
            "edge_tts": self._edge_tts_available,
            "pyttsx3_fallback": self._pyttsx3_available,
            "default_voice": TTS_CONFIG["default_voice"],
        }


# Global singleton instance
_tts_service: Optional[TTSService] = None


def get_tts_service(preload: bool = False) -> TTSService:
    """
    Get or create the global TTS service instance.
    
    Args:
        preload: If True and creating new instance, preload models
    """
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService(preload_models=preload)
    return _tts_service


def preload_tts_model():
    """Preload TTS model. Call during application startup."""
    service = get_tts_service()
    service.preload()
    logger.info("TTS model preloaded")
