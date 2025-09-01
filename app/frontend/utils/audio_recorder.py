"""
Audio recorder utilities with multi-backend support.
"""
import logging
from typing import Optional, Callable, Any

logger = logging.getLogger(__name__)

# Global state for audio recording
AUD_RECORDER_BACKEND: Optional[str] = None
AUD_IMPORT_ERROR: Optional[Exception] = None
record_audio: Optional[Callable] = None


def _init_audio_backend() -> None:
    """Initialize audio recording backend with fallback options."""
    global AUD_RECORDER_BACKEND, AUD_IMPORT_ERROR, record_audio
    
    # Attempt 1: streamlit_audiorecorder ("streamlit-audiorecorder")
    try:
        import streamlit_audiorecorder as _sar  # type: ignore
        def record_audio_impl(**kwargs):  # wrapper to keep a consistent API
            return _sar.audio_recorder(**kwargs)
        record_audio = record_audio_impl
        AUD_RECORDER_BACKEND = "streamlit_audiorecorder"
        return
    except Exception as e:
        AUD_IMPORT_ERROR = e
        logger.debug(f"Failed to import streamlit_audiorecorder: {e}")
    
    # Attempt 2: audio_recorder_streamlit ("audio-recorder-streamlit")
    try:
        from audio_recorder_streamlit import audio_recorder as _ars  # type: ignore
        def record_audio_impl(**kwargs):
            # This variant uses different arg names; map the most common ones.
            text = kwargs.get("text", "")
            recording_color = kwargs.get("recording_color", "red")
            neutral_color = kwargs.get("neutral_color", "gray")
            # Component returns raw wav bytes directly.
            return _ars(text, text, recording_color=recording_color, neutral_color=neutral_color)
        record_audio = record_audio_impl
        AUD_RECORDER_BACKEND = "audio_recorder_streamlit"
        return
    except Exception as e:
        AUD_IMPORT_ERROR = e
        logger.debug(f"Failed to import audio_recorder_streamlit: {e}")
    
    # Attempt 3: audiorecorder ("streamlit-audiorecorder" older API)
    try:
        from audiorecorder import audiorecorder as _legacy  # type: ignore
        def record_audio_impl(**kwargs):
            # Returns a pydub AudioSegment; convert to bytes via export.
            seg = _legacy("Click to record", "Click to stop")
            if seg and hasattr(seg, 'export'):
                import io
                buff = io.BytesIO()
                seg.export(buff, format="wav")
                return buff.getvalue()
            return None
        record_audio = record_audio_impl
        AUD_RECORDER_BACKEND = "audiorecorder_legacy"
        return
    except Exception as e:
        AUD_IMPORT_ERROR = e
        logger.debug(f"Failed to import audiorecorder: {e}")


# Initialize backend on import
_init_audio_backend()

# Export status
AUDIO_RECORDER_AVAILABLE = AUD_RECORDER_BACKEND is not None


def get_recorder_info() -> dict:
    """Get information about the audio recorder backend."""
    return {
        "available": AUDIO_RECORDER_AVAILABLE,
        "backend": AUD_RECORDER_BACKEND,
        "error": str(AUD_IMPORT_ERROR) if AUD_IMPORT_ERROR else None
    }
