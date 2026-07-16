"""Configuration dataclasses and constants for realtime voice chat.

This module is dependency-free (stdlib only) so it can be imported by every
other module in the package without risk of circular imports or triggering
heavy third-party imports.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass


@dataclass
class _VadConfig:
    """Configuration for webrtcvad fallback."""

    aggressiveness: int = 2
    frame_ms: int = 20
    silence_frames_to_finalize: int = 40  # ~800ms at 20ms frames (increased from 15)
    min_speech_frames: int = 8  # ~160ms


@dataclass
class _PipecatConfig:
    """Configuration for Pipecat VAD."""

    frame_ms: int = 20
    segment_sample_rate: int = 16000
    pre_roll_ms: int = 200
    min_utterance_ms: int = 300
    # VAD parameters - increased stop_secs for natural pauses
    vad_start_secs: float = 0.2
    vad_stop_secs: float = 1.5  # Increased from 0.8 to allow natural pauses
    vad_confidence: float = 0.65  # Slightly lower for more natural detection
    vad_min_volume: float = 0.05  # Standard sensitive volume threshold (default 0.6 is too high)


# Default instances. These are never mutated, so a single shared instance is
# safe and avoids repeating the default values at every call site.
VAD_CONFIG = _VadConfig()
PIPECAT_CONFIG = _PipecatConfig()

# Sentence-ending punctuation for streaming TTS
SENTENCE_ENDINGS = re.compile(r"[.!?;:]\s*")
MIN_TTS_CHUNK_SIZE = 15  # Reduced for faster first audio (was 30)

# Greeting message for when realtime voice starts
REALTIME_GREETING_CHAT = "Good day! How can I help you today?"
REALTIME_GREETING_RAG = "Good day! Ask me anything about your documents."

# Thread-safe cache for greeting audio (not using Streamlit session state)
# This avoids ScriptRunContext warnings when caching from background threads
_greeting_audio_cache: dict = {}
_greeting_cache_lock = threading.Lock()
GREETING_CACHE_KEY = "realtime_greeting_audio_cached"
