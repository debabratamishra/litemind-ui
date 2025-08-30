"""
Speech-to-Text service using Hugging Face Whisper model.

This module provides functionality to transcribe audio files to text using
an open-source Whisper model from Hugging Face.
"""

import logging
import tempfile
from typing import Optional

import librosa
import numpy as np
import torch
from transformers import pipeline

logger = logging.getLogger(__name__)

class SpeechService:
    """Service for speech-to-text transcription using Whisper."""

    def __init__(self, model_name: str = "openai/whisper-small"):
        """
        Initialize the speech service with a Whisper model.

        Args:
            model_name: Hugging Face model name for Whisper
        """
        self.model_name = model_name
        self.pipe = None
        self._load_model()

    def _load_model(self):
        """Load the Whisper model pipeline."""
        try:
            logger.info(f"Loading Whisper model: {self.model_name}")
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def transcribe_audio(self, audio_data: bytes, sample_rate: int = 16000) -> Optional[str]:
        """
        Transcribe audio data to text.

        Args:
            audio_data: Raw audio bytes from streamlit-audiorecorder
            sample_rate: Sample rate of the audio

        Returns:
            Transcribed text or None if transcription fails
        """
        try:
            import io, soundfile as sf
            # Primary attempt: librosa direct load
            try:
                audio_array, sr = librosa.load(io.BytesIO(audio_data), sr=sample_rate)
            except Exception:
                # Fallback: write to temp file then load
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                    tmp.write(audio_data)
                    tmp.flush()
                    audio_array, sr = librosa.load(tmp.name, sr=sample_rate)

            # Ensure float32
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)

            # Whisper pipeline accepts raw array
            result = self.pipe(audio_array)
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
        try:
            # Load audio file
            audio_array, sample_rate = librosa.load(file_path, sr=16000)

            # Transcribe
            result = self.pipe(audio_array)
            text = result["text"].strip()

            logger.info(f"File transcription successful: {len(text)} characters")
            return text

        except Exception as e:
            logger.error(f"File transcription failed: {e}")
            return None


# Global instance
_speech_service: Optional[SpeechService] = None

def get_speech_service() -> SpeechService:
    """Get or create the global speech service instance."""
    global _speech_service
    if _speech_service is None:
        _speech_service = SpeechService()
    return _speech_service
