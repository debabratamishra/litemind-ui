"""
Speech-to-Text service using Hugging Face Whisper model.

This module provides functionality to transcribe audio files to text using
an open-source Whisper model from Hugging Face transformers library.

Features:
- Model preloading: Load models at startup for reduced latency
- Uses transformers pipeline for Whisper STT
- Streaming transcription: Get partial results during speech
- Configurable via environment variables
"""

import logging
import os
import tempfile
import threading
import warnings
from typing import Optional, Generator, Tuple, Callable

import librosa
import numpy as np
import torch

# Suppress the FutureWarning from transformers about 'inputs' deprecation
# This is an internal transformers issue that will be fixed in a future version
warnings.filterwarnings(
    "ignore", 
    message=".*input name `inputs` is deprecated.*",
    category=FutureWarning,
    module="transformers.*"
)

from transformers import pipeline

logger = logging.getLogger(__name__)


class SpeechService:
    """Service for speech-to-text transcription using Whisper."""

    def __init__(self, model_name: str = "openai/whisper-base.en", preload: bool = False):
        """
        Initialize the speech service with a Whisper model.

        Args:
            model_name: Hugging Face model name for Whisper
            preload: If True, load the model immediately during init
        """
        self.model_name = model_name
        self.backend = "transformers"  # Only transformers backend supported now
        self.pipe = None
        self._model_loaded = False
        
        if preload:
            self._load_model()

    def _load_model(self):
        """Load the Whisper model pipeline."""
        if self._model_loaded:
            return
            
        try:
            logger.info(f"Loading transformers Whisper model: {self.model_name}")
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                # Enable return_timestamps for long-form audio (>30s)
                return_timestamps=True,
            )
            self._model_loaded = True
            logger.info("Transformers Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def preload(self):
        """
        Public method to preload the model.
        Call during application startup to reduce first-request latency.
        """
        if not self._model_loaded:
            logger.info("Preloading speech-to-text model...")
            self._load_model()
            logger.info("Speech-to-text model preloaded successfully")

    def is_model_loaded(self) -> bool:
        """Check if the STT model is loaded in memory."""
        return self._model_loaded

    def _ensure_model_loaded(self):
        """Ensure the model is loaded before transcription."""
        if not self._model_loaded:
            self._load_model()

    def transcribe_audio(self, audio_data: bytes, sample_rate: int = 16000) -> Optional[str]:
        """
        Transcribe audio data to text.

        Args:
            audio_data: Raw audio bytes from streamlit-audiorecorder
            sample_rate: Sample rate of the audio

        Returns:
            Transcribed text or None if transcription fails
        """
        self._ensure_model_loaded()
        
        try:
            import io
            try:
                audio_array, sr = librosa.load(io.BytesIO(audio_data), sr=sample_rate)
            except Exception:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                    tmp.write(audio_data)
                    tmp.flush()
                    audio_array, sr = librosa.load(tmp.name, sr=sample_rate)

            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)

            # transformers pipeline accepts raw array
            if self.pipe is None:
                raise RuntimeError("SpeechService pipeline not initialized")
            # Pass dict with 'raw' key to avoid deprecation warning
            result = self.pipe({"raw": audio_array, "sampling_rate": sample_rate})
            text = (result.get("text") or "").strip()

            if not text:
                logger.warning("Transcription returned empty text")
                return None
            logger.info(f"Transcription successful: {len(text)} characters")
            return text

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return None

    def transcribe_file(self, file_path: str) -> Optional[str]:
        """
        Transcribe an audio file to text.

        Args:
            file_path: Path to the audio file

        Returns:
            Transcribed text or None if transcription fails
        """
        self._ensure_model_loaded()
        
        try:
            # Load audio file
            audio_array, _sample_rate = librosa.load(file_path, sr=16000)
            if self.pipe is None:
                raise RuntimeError("SpeechService pipeline not initialized")

            # Transcribe - use dict with 'raw' key to avoid deprecation warning
            result = self.pipe({"raw": audio_array, "sampling_rate": 16000})
            text = (result.get("text") or "").strip()

            logger.info(f"File transcription successful: {len(text)} characters")
            return text

        except Exception as e:
            logger.error(f"File transcription failed: {e}")
            return None

    def transcribe_audio_streaming(
        self, 
        audio_data: bytes, 
        sample_rate: int = 16000,
        on_partial: Optional[Callable[[str], None]] = None
    ) -> Optional[str]:
        """
        Transcribe audio with streaming partial results.
        
        This method provides intermediate transcription results as they become
        available, which is useful for showing live transcription in the UI.
        
        Args:
            audio_data: Raw audio bytes
            sample_rate: Sample rate of the audio
            on_partial: Callback function called with partial transcription text
            
        Returns:
            Final transcribed text or None if transcription fails
        """
        self._ensure_model_loaded()
        
        try:
            import io
            try:
                audio_array, sr = librosa.load(io.BytesIO(audio_data), sr=sample_rate)
            except Exception:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                    tmp.write(audio_data)
                    tmp.flush()
                    audio_array, sr = librosa.load(tmp.name, sr=sample_rate)

            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)

            # transformers pipeline - single result (no streaming support)
            if self.pipe is None:
                raise RuntimeError("SpeechService pipeline not initialized")
            
            # Show "Processing..." while transcribing
            if on_partial:
                on_partial("Processing...")
            
            # Use dict with 'raw' key to avoid deprecation warning
            result = self.pipe({"raw": audio_array, "sampling_rate": sample_rate})
            text = (result.get("text") or "").strip()
            
            # Send final result
            if on_partial and text:
                on_partial(text)

            if not text:
                logger.warning("Streaming transcription returned empty text")
                return None
            logger.info(f"Streaming transcription successful: {len(text)} characters")
            return text

        except Exception as e:
            logger.error(f"Streaming transcription failed: {e}")
            return None

    def transcribe_chunk_generator(
        self, 
        audio_data: bytes, 
        sample_rate: int = 16000
    ) -> Generator[Tuple[str, bool], None, None]:
        """
        Generator that yields transcription chunks.
        
        Yields:
            Tuple of (text, is_final) where is_final indicates if this is the final result
        """
        self._ensure_model_loaded()
        
        try:
            import io
            try:
                audio_array, sr = librosa.load(io.BytesIO(audio_data), sr=sample_rate)
            except Exception:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                    tmp.write(audio_data)
                    tmp.flush()
                    audio_array, sr = librosa.load(tmp.name, sr=sample_rate)

            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)

            # transformers pipeline - single result
            if self.pipe is None:
                raise RuntimeError("SpeechService pipeline not initialized")
            
            yield ("Processing...", False)
            # Use dict with 'raw' key to avoid deprecation warning
            result = self.pipe({"raw": audio_array, "sampling_rate": sample_rate})
            text = (result.get("text") or "").strip()
            yield (text, True)

        except Exception as e:
            logger.error(f"Transcription generator failed: {e}")
            yield (f"Error: {e}", True)
            return None

    def get_status(self) -> dict:
        """Get speech service status."""
        return {
            "available": True,
            "model_loaded": self._model_loaded,
            "backend": self.backend,
            "model_name": self.model_name,
        }


# Global instance
_speech_service: Optional[SpeechService] = None


def get_speech_service(preload: bool = False) -> SpeechService:
    """
    Get or create the global speech service instance.
    
    Args:
        preload: If True and creating new instance, preload the model
    """
    global _speech_service
    if _speech_service is None:
        _speech_service = SpeechService(preload=preload)
    return _speech_service


def preload_stt_model():
    """Preload STT model. Call during application startup."""
    service = get_speech_service()
    service.preload()
    logger.info("STT model preloaded")
