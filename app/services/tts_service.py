"""
Text-to-Speech service using lightweight open-source models.

This module provides TTS functionality using:
1. Kokoro TTS (High quality, offline, fast) - Preferred
2. Edge TTS (Microsoft Edge's TTS - fast, high quality, no API key needed)
3. pyttsx3 as fallback (offline, system voices)

Kokoro is preferred as it's offline, high quality, and supports streaming.
"""

import asyncio
import io
import logging
import os
import re
import tempfile
from typing import Optional, Tuple
import hashlib

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


class TTSService:
    """Text-to-Speech service with multiple backend support."""
    
    def __init__(self):
        self._preferred_backend = os.getenv("TTS_BACKEND", "auto").strip().lower()
        self._edge_tts_available = False
        self._pyttsx3_available = False
        self._kokoro_available = False
        self._edge_tts = None
        self._pyttsx3_engine = None
        self._kokoro_pipeline = None
        self._check_backends()
        self._ensure_cache_dir()
    
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
            # Initialize pipeline lazily or here? 
            # Initializing here might block startup, but ensures it's ready.
            # Let's initialize lazily in _synthesize_kokoro to avoid startup delay.
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
    
    def _clean_text_for_tts(self, text: str) -> str:
        """Clean text for TTS by removing markdown and special formatting."""
        # Remove thinking/reasoning tags
        text = re.sub(r'<\s*(think|thinking|reasoning|thought)\s*>.*?<\s*/\s*(think|thinking|reasoning|thought)\s*>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove markdown code blocks
        text = re.sub(r'```[\s\S]*?```', ' code block omitted ', text)
        text = re.sub(r'`[^`]+`', '', text)
        
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
        
        # Clean up extra whitespace
        text = re.sub(r'\n\s*\n', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
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
                # 'a' for American English. 
                # TODO: Detect language or allow config.
                self._kokoro_pipeline = KPipeline(lang_code='a')
            
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
            # Try one retry with a different voice if it was the default voice
            if voice == TTS_CONFIG["default_voice"] and TTS_CONFIG["alternative_voices"]:
                alt_voice = TTS_CONFIG["alternative_voices"][0]
                logger.info(f"Retrying Edge TTS with alternative voice: {alt_voice}")
                try:
                    communicate = self._edge_tts.Communicate(
                        text,
                        alt_voice,
                        rate=TTS_CONFIG["rate"],
                        volume=TTS_CONFIG["volume"],
                        pitch=TTS_CONFIG["pitch"]
                    )
                    audio_data = io.BytesIO()
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            audio_data.write(chunk["data"])
                    result = audio_data.getvalue()
                    if result:
                        logger.info(f"Edge TTS retry success with {alt_voice}")
                        return result
                except Exception as retry_e:
                    logger.error(f"Edge TTS retry failed: {retry_e}")
            
            return None
    
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
        return self._edge_tts_available or self._pyttsx3_available
    
    def get_status(self) -> dict:
        """Get TTS service status."""
        return {
            "available": self.is_available(),
            "preferred_backend": self._preferred_backend,
            "kokoro": self._kokoro_available,
            "edge_tts": self._edge_tts_available,
            "pyttsx3_fallback": self._pyttsx3_available,
            "default_voice": TTS_CONFIG["default_voice"],
        }


# Global singleton instance
_tts_service: Optional[TTSService] = None


def get_tts_service() -> TTSService:
    """Get or create the global TTS service instance."""
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService()
    return _tts_service
