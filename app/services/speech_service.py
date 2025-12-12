"""
Speech-to-Text service using Hugging Face Whisper model.

This module provides functionality to transcribe audio files to text using
an open-source Whisper model from Hugging Face.
"""

import logging
import os
import tempfile
from typing import Optional

import librosa
import numpy as np
import torch
from transformers import pipeline

logger = logging.getLogger(__name__)

class SpeechService:
    """Service for speech-to-text transcription using Whisper."""

    def __init__(self, model_name: str = "openai/whisper-base.en"):
        """
        Initialize the speech service with a Whisper model.

        Args:
            model_name: Hugging Face model name for Whisper
        """
        self.model_name = model_name
        self.backend = os.getenv("STT_BACKEND", "transformers").strip().lower()
        self.pipe = None
        self._fw_model = None
        self._load_model()

    def _load_model(self):
        """Load the Whisper model pipeline."""
        try:
            if self.backend in {"faster-whisper", "fastwhisper", "fast-whisper"}:
                from faster_whisper import WhisperModel  # type: ignore

                fw_model = os.getenv("FASTWHISPER_MODEL", "base.en").strip()
                fw_device = os.getenv("FASTWHISPER_DEVICE", "cpu").strip().lower()
                fw_compute_type = os.getenv("FASTWHISPER_COMPUTE_TYPE", "int8").strip().lower()

                logger.info(
                    "Loading faster-whisper model: model=%s device=%s compute_type=%s",
                    fw_model,
                    fw_device,
                    fw_compute_type,
                )
                self._fw_model = WhisperModel(fw_model, device=fw_device, compute_type=fw_compute_type)
                self.pipe = None
                self.backend = "faster-whisper"
                logger.info("faster-whisper model loaded successfully")
                return

            logger.info(f"Loading transformers Whisper model: {self.model_name}")
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            self._fw_model = None
            self.backend = "transformers"
            logger.info("Transformers Whisper model loaded successfully")
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

            # faster-whisper path
            if self.backend == "faster-whisper" and self._fw_model is not None:
                language = os.getenv("FASTWHISPER_LANGUAGE", "en").strip() or None
                vad_filter = os.getenv("FASTWHISPER_VAD", "1").strip() not in {"0", "false", "False"}
                beam_size = int(os.getenv("FASTWHISPER_BEAM_SIZE", "1"))

                segments, _info = self._fw_model.transcribe(
                    audio_array,
                    language=language,
                    vad_filter=vad_filter,
                    beam_size=beam_size,
                )
                text = "".join((seg.text or "") for seg in segments).strip()
            else:
                # transformers pipeline accepts raw array
                if self.pipe is None:
                    raise RuntimeError("SpeechService pipeline not initialized")
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
            if self.backend == "faster-whisper" and self._fw_model is not None:
                language = os.getenv("FASTWHISPER_LANGUAGE", "en").strip() or None
                vad_filter = os.getenv("FASTWHISPER_VAD", "1").strip() not in {"0", "false", "False"}
                beam_size = int(os.getenv("FASTWHISPER_BEAM_SIZE", "1"))
                segments, _info = self._fw_model.transcribe(
                    file_path,
                    language=language,
                    vad_filter=vad_filter,
                    beam_size=beam_size,
                )
                text = "".join((seg.text or "") for seg in segments).strip()
            else:
                # Load audio file
                audio_array, _sample_rate = librosa.load(file_path, sr=16000)
                if self.pipe is None:
                    raise RuntimeError("SpeechService pipeline not initialized")

                # Transcribe
                result = self.pipe(audio_array)
                text = (result.get("text") or "").strip()

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
