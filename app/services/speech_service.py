"""
Speech-to-Text service using Hugging Face Whisper model.

This module provides functionality to transcribe audio files to text using
an open-source Whisper model from Hugging Face.

Features:
- Model preloading: Load models at startup for reduced latency
- Multiple backends: transformers pipeline or faster-whisper
- Streaming transcription: Get partial results during speech
- Configurable via environment variables
"""

import logging
import os
import tempfile
import threading
from typing import Optional, Generator, Tuple, Callable

import librosa
import numpy as np
import torch
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
        self.backend = os.getenv("STT_BACKEND", "transformers").strip().lower()
        self.pipe = None
        self._fw_model = None
        self._model_loaded = False
        
        if preload:
            self._load_model()

    def _load_model(self):
        """Load the Whisper model pipeline."""
        if self._model_loaded:
            return
            
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
                self._model_loaded = True
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
        self._ensure_model_loaded()
        
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

            # faster-whisper supports streaming segments
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
                
                # Stream segments as they come
                text_parts = []
                for seg in segments:
                    seg_text = (seg.text or "").strip()
                    if seg_text:
                        text_parts.append(seg_text)
                        partial_text = " ".join(text_parts)
                        if on_partial:
                            try:
                                on_partial(partial_text)
                            except Exception as e:
                                logger.debug(f"Partial callback error: {e}")
                
                text = " ".join(text_parts).strip()
            else:
                # transformers pipeline - single result (no streaming support)
                if self.pipe is None:
                    raise RuntimeError("SpeechService pipeline not initialized")
                
                # Show "Processing..." while transcribing
                if on_partial:
                    on_partial("Processing...")
                
                result = self.pipe(audio_array)
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

            # faster-whisper supports streaming segments
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
                
                # Yield segments as they come
                text_parts = []
                segment_list = list(segments)  # Consume generator
                
                for i, seg in enumerate(segment_list):
                    seg_text = (seg.text or "").strip()
                    if seg_text:
                        text_parts.append(seg_text)
                        partial_text = " ".join(text_parts)
                        is_final = (i == len(segment_list) - 1)
                        yield (partial_text, is_final)
                
                # If no segments, yield empty final
                if not text_parts:
                    yield ("", True)
            else:
                # transformers pipeline - single result
                if self.pipe is None:
                    raise RuntimeError("SpeechService pipeline not initialized")
                
                yield ("Processing...", False)
                result = self.pipe(audio_array)
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
