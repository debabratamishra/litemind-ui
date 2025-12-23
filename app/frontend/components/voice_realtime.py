"""Realtime voice chat component (WebRTC).

This provides a "speech → LLM → speech" loop inside the Streamlit UI.

Features:
- Captures microphone audio via WebRTC (streamlit-webrtc)
- Segments speech using Pipecat Silero VAD (preferred) or webrtcvad fallback
- Transcribes with the existing SpeechService (optionally faster-whisper backend)
- Streams an LLM reply using the existing streaming_handler (Ollama/vLLM)
- Synthesizes response audio via backend TTS endpoint with streaming support
- Reduced latency through sentence-by-sentence TTS synthesis

Architecture:
- PipecatVADProcessor: Uses Pipecat Silero VAD for better speech detection
- WebRTCVADProcessor: Fallback using webrtcvad
- StreamingTTSHandler: Synthesizes audio as LLM text streams in
"""

from __future__ import annotations

import io
import logging
import queue
import re
import threading
import time
import concurrent.futures
from dataclasses import dataclass
from typing import Any, Optional, Tuple, List, Callable

import numpy as np
import requests
import streamlit as st

from ..config import FASTAPI_URL
from ..components.streaming_handler import streaming_handler
from ...services.speech_service import get_speech_service

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

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


# Sentence-ending punctuation for streaming TTS
SENTENCE_ENDINGS = re.compile(r'[.!?;:]\s*')
MIN_TTS_CHUNK_SIZE = 15  # Reduced for faster first audio (was 30)

# Greeting message for when realtime voice starts
REALTIME_GREETING_CHAT = "Good day! How can I help you today?"
REALTIME_GREETING_RAG = "Good day! Ask me anything about your documents."
GREETING_CACHE_KEY = "realtime_greeting_audio_cached"


# ============================================================================
# Utility Functions
# ============================================================================

def _pcm16_to_wav_bytes(pcm16: bytes, sample_rate: int, channels: int = 1) -> bytes:
    """Convert raw PCM16 bytes to WAV format."""
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16)
    return buf.getvalue()


def _get_realtime_greeting(page_key: str) -> str:
    """Get the appropriate greeting message based on page."""
    if page_key == "rag":
        return REALTIME_GREETING_RAG
    return REALTIME_GREETING_CHAT


def _get_chat_config_from_session(page_key: str = "chat") -> dict:
    """Extract chat configuration from session state.
    
    Args:
        page_key: The page key ("chat" or "rag") to determine appropriate model selection.
    """
    backend_provider = st.session_state.get("current_backend", "ollama")
    temperature = st.session_state.get("chat_temperature", 0.7)
    if backend_provider == "vllm":
        model = st.session_state.get("vllm_model", "no-model")
        hf_token = st.session_state.get("hf_token")
    else:
        # Use appropriate model key based on page
        if page_key == "rag":
            model = st.session_state.get("selected_ollama_model", "default")
        else:
            model = st.session_state.get("selected_chat_model", "default")
        hf_token = None

    return {
        "backend": backend_provider,
        "model": model,
        "temperature": temperature,
        "hf_token": hf_token,
    }


def _get_messages_key(page_key: str) -> str:
    """Get the session state key for messages based on page."""
    if page_key == "rag":
        return "rag_messages"
    return "chat_messages"


def _decode_audio_bytes_to_pcm16_mono(
    audio_bytes: bytes, *, target_sample_rate: int
) -> Tuple[bytes, int]:
    """Decode common audio formats (mp3/wav) into PCM16 mono at target_sample_rate."""
    try:
        import av
        from av.audio.resampler import AudioResampler
    except ImportError as e:
        raise RuntimeError("PyAV (av) is required to decode TTS audio") from e

    if not audio_bytes:
        return b"", target_sample_rate

    try:
        container = av.open(io.BytesIO(audio_bytes))
    except Exception as e:
        logger.warning(f"PyAV failed to open audio bytes: {e}")
        return b"", target_sample_rate

    audio_streams = [s for s in container.streams if s.type == "audio"]
    if not audio_streams:
        return b"", target_sample_rate

    in_stream = audio_streams[0]
    resampler = AudioResampler(
        format="s16",
        layout="mono",
        rate=target_sample_rate,
    )

    out = bytearray()
    try:
        for packet in container.demux(in_stream):
            for frame in packet.decode():
                for out_frame in resampler.resample(frame):
                    pcm = out_frame.to_ndarray()
                    if pcm.ndim == 2:
                        pcm = pcm[0]
                    if pcm.dtype != np.int16:
                        pcm = np.clip(pcm, -32768, 32767).astype(np.int16)
                    out.extend(pcm.tobytes())
    except Exception as e:
        logger.warning(f"PyAV decoding error: {e}")
        return b"", target_sample_rate

    return bytes(out), target_sample_rate


# ============================================================================
# Streaming TTS Handler
# ============================================================================

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
        self._synthesized_sentences: List[str] = []
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
    
    Args:
        page_key: The page key ("chat" or "rag") to determine appropriate greeting.
    """
    cache_key = f"{GREETING_CACHE_KEY}_{page_key}"
    greeting_text = _get_realtime_greeting(page_key)
    
    # Check if we already have cached greeting in session state
    if cache_key in st.session_state:
        cached = st.session_state[cache_key]
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
            st.session_state[cache_key] = resp.content
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
        pcm_out, _ = _decode_audio_bytes_to_pcm16_mono(
            audio_bytes, target_sample_rate=target_sr
        )
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
        if audio_bytes[:4] == b'RIFF':
            fmt = "audio/wav"
        
        st.audio(io.BytesIO(audio_bytes), format=fmt, autoplay=True)
        logger.info("Greeting played via audio widget")
        return True
    except Exception as e:
        logger.error(f"Failed to play greeting via audio widget: {e}")
    
    return False


# ============================================================================
# Audio Output Queue (for barge-in support)
# ============================================================================

class _InterruptibleAudioOut:
    """Thread-safe audio output queue that supports interruption (barge-in)."""
    
    def __init__(self) -> None:
        self._q: queue.Queue[bytes] = queue.Queue()
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._bot_speaking = False
        self._interrupted = threading.Event()  # Signal for barge-in
        self._interrupt_callback: Optional[Callable[[], None]] = None

    def set_interrupt_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """Set a callback to be called when interrupt occurs (for TTS handler)."""
        self._interrupt_callback = callback

    def enqueue(self, pcm16: bytes) -> None:
        """Add audio to the output queue."""
        if not pcm16 or self._interrupted.is_set():
            return
        with self._lock:
            self._bot_speaking = True
            self._q.put(pcm16)

    def interrupt(self) -> None:
        """Clear all queued audio (user started speaking)."""
        self._interrupted.set()  # Signal interruption
        with self._lock:
            self._bot_speaking = False
            self._buf.clear()
            try:
                while True:
                    self._q.get_nowait()
            except queue.Empty:
                pass
        # Call the interrupt callback if set (to stop TTS synthesis)
        if self._interrupt_callback:
            try:
                self._interrupt_callback()
            except Exception:
                pass

    def reset_interrupt(self) -> None:
        """Reset the interrupt flag for a new response."""
        self._interrupted.clear()

    def is_interrupted(self) -> bool:
        """Check if audio output has been interrupted."""
        return self._interrupted.is_set()

    def is_bot_speaking(self) -> bool:
        """Check if there's audio being output."""
        with self._lock:
            return self._bot_speaking and (len(self._buf) > 0 or not self._q.empty())

    def get_bytes_for_frame(self, nbytes: int) -> bytes:
        """Get bytes for output frame, returns silence if not enough."""
        if nbytes <= 0:
            return b""
        with self._lock:
            while len(self._buf) < nbytes:
                try:
                    chunk = self._q.get_nowait()
                except queue.Empty:
                    break
                self._buf.extend(chunk)

            if len(self._buf) >= nbytes:
                out = bytes(self._buf[:nbytes])
                del self._buf[:nbytes]
            else:
                out = b""

            if len(self._buf) == 0 and self._q.empty():
                self._bot_speaking = False
            return out


# ============================================================================
# Main Render Function
# ============================================================================

def render_realtime_voice_chat(page_key: str = "chat") -> None:
    """Render realtime voice chat UI.

    This function manages its own session-state and writes to the shared
    `st.session_state.chat_messages` history for continuity with the Chat page.
    """
    realtime_mode_key = f"realtime_voice_mode_{page_key}"

    # Check for required packages
    try:
        from streamlit_webrtc import AudioProcessorBase, WebRtcMode, webrtc_streamer
        import av
    except ImportError as e:
        st.warning(
            "Realtime voice chat requires additional packages. "
            "Install `streamlit-webrtc` and `av` in your environment."
        )
        st.code(str(e))
        return

    # Check for webrtcvad (fallback)
    try:
        import webrtcvad
        _WEBRTCVAD_AVAILABLE = True
    except ImportError:
        webrtcvad = None
        _WEBRTCVAD_AVAILABLE = False
        logger.warning("webrtcvad not available")

    # Check for Pipecat
    try:
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.audio.vad.vad_analyzer import VADParams, VADState
        _PIPECAT_AVAILABLE = True
    except ImportError:
        SileroVADAnalyzer = None
        VADParams = None
        VADState = None
        _PIPECAT_AVAILABLE = False

    if not _PIPECAT_AVAILABLE and not _WEBRTCVAD_AVAILABLE:
        st.error("No VAD backend available. Install pipecat-ai[silero] or webrtcvad.")
        return

    # Configuration
    vad_cfg = _VadConfig()
    pipecat_cfg = _PipecatConfig()

    # CSS for the modern circle UI
    st.markdown("""
        <style>
        .voice-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 2rem;
        }
        .voice-circle {
            width: 200px;
            height: 200px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            box-shadow: 0 0 30px rgba(118, 75, 162, 0.6);
            display: flex;
            align-items: center;
            justify-content: center;
            animation: pulse 3s infinite ease-in-out;
            margin-bottom: 2rem;
            color: white;
            font-size: 4rem;
        }
        @keyframes pulse {
            0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(118, 75, 162, 0.7); }
            50% { transform: scale(1.05); box-shadow: 0 0 0 20px rgba(118, 75, 162, 0); }
            100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(118, 75, 162, 0); }
        }
        .voice-status {
            font-size: 1.5rem;
            font-weight: 600;
            color: #555;
            margin-bottom: 1rem;
        }
        .voice-subtext {
            font-size: 1rem;
            color: #888;
        }
        </style>
    """, unsafe_allow_html=True)

    # Close button
    col_spacer, col_close = st.columns([10, 1])
    with col_close:
        if st.button("✕", key=f"realtime_voice_close_{page_key}", help="Close realtime voice chat"):
            # Save any ongoing partial response before closing
            active_tts_key = f"active_tts_handler_{page_key}"
            messages_key = _get_messages_key(page_key)
            
            if active_tts_key in st.session_state and st.session_state[active_tts_key]:
                try:
                    tts_handler = st.session_state[active_tts_key]
                    # Interrupt the TTS handler
                    tts_handler.interrupt()
                    
                    # Get any partial text that was received
                    partial_text = tts_handler.get_full_text()
                    if partial_text and partial_text.strip():
                        # Check if we already have this response in history (avoid duplicates)
                        messages = st.session_state.get(messages_key, [])
                        if not messages or messages[-1].get("role") != "assistant" or messages[-1].get("content") != partial_text.strip():
                            st.session_state[messages_key].append({
                                "role": "assistant", 
                                "content": partial_text.strip() + " [interrupted]"
                            })
                            logger.info(f"Saved interrupted response: {len(partial_text)} chars")
                    
                    tts_handler.shutdown()
                except Exception as e:
                    logger.debug(f"Error saving partial response on close: {e}")
                
                st.session_state[active_tts_key] = None
            
            # Clear live transcript
            live_transcript_key = f"live_transcript_{page_key}"
            if live_transcript_key in st.session_state:
                st.session_state[live_transcript_key] = ""
            
            st.session_state[realtime_mode_key] = False
            st.rerun()

    # Visual Indicator
    page_label = "RAG Document Assistant" if page_key == "rag" else "Chat Assistant"
    page_subtext = "Ask me anything about your documents." if page_key == "rag" else "Speak naturally. I will listen and respond."
    
    st.markdown('<div class="voice-container">', unsafe_allow_html=True)
    st.markdown('<div class="voice-circle">🎙️</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="voice-status">{page_label}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="voice-subtext">{page_subtext}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


    # ========================================================================
    # Audio Processor: Pipecat VAD (preferred)
    # ========================================================================
    
    class PipecatVADProcessor(AudioProcessorBase):
        """Audio processor using Pipecat Silero VAD for speech detection."""
        
        def __init__(self) -> None:
            self._segments: queue.Queue[Tuple[bytes, int]] = queue.Queue()
            self._interim_audio: queue.Queue[Tuple[bytes, int]] = queue.Queue()  # For live transcription
            self._stop = threading.Event()
            self._out = _InterruptibleAudioOut()
            
            # Audio state
            self._io_sample_rate: Optional[int] = None
            self._last_frame_format: Optional[Tuple] = None
            self._resampler: Optional[Any] = None
            
            # VAD state (thread-safe)
            self._vad_lock = threading.Lock()
            self._vad: Optional[Any] = None
            self._collecting = False
            self._utterance = bytearray()
            self._pre_roll = bytearray()
            self._in_speech = False
            self._silence_frames = 0
            self._interim_counter = 0  # Counter for interim updates
            
            # Initialize VAD
            self._init_vad()
        
        def _init_vad(self) -> None:
            """Initialize Pipecat Silero VAD."""
            if not _PIPECAT_AVAILABLE or SileroVADAnalyzer is None:
                return
            try:
                self._vad = SileroVADAnalyzer(
                    params=VADParams(
                        confidence=pipecat_cfg.vad_confidence,
                        start_secs=pipecat_cfg.vad_start_secs,
                        stop_secs=pipecat_cfg.vad_stop_secs,
                    )
                )
                self._vad.set_sample_rate(pipecat_cfg.segment_sample_rate)
                logger.info("Pipecat Silero VAD initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Pipecat VAD: {e}")
                self._vad = None
        
        @property
        def segments(self) -> queue.Queue[Tuple[bytes, int]]:
            return self._segments
        
        @property
        def output_sample_rate(self) -> int:
            return int(self._io_sample_rate or 48000)
        
        def enqueue_assistant_pcm16(self, pcm16: bytes) -> None:
            self._out.enqueue(pcm16)
        
        def interrupt_assistant(self) -> None:
            self._out.interrupt()
        
        def set_tts_interrupt_callback(self, callback: Optional[Callable[[], None]]) -> None:
            """Set a callback to be invoked on barge-in (to stop TTS handler)."""
            self._out.set_interrupt_callback(callback)
        
        def reset_interrupt(self) -> None:
            """Reset the interrupt flag for a new response."""
            self._out.reset_interrupt()
        
        def is_audio_interrupted(self) -> bool:
            """Check if audio output has been interrupted (barge-in occurred)."""
            return self._out.is_interrupted()
        
        def _get_resampler(self, frame) -> Optional[Any]:
            """Get or create a resampler for the current frame format."""
            from av.audio.resampler import AudioResampler
            
            frame_format = (
                str(frame.format.name) if frame.format else None,
                str(frame.layout.name) if frame.layout else None,
                frame.sample_rate,
            )
            
            if self._resampler is None or self._last_frame_format != frame_format:
                try:
                    self._resampler = AudioResampler(
                        format="s16",
                        layout="mono",
                        rate=pipecat_cfg.segment_sample_rate,
                    )
                    self._last_frame_format = frame_format
                except Exception as e:
                    logger.warning(f"Failed to create resampler: {e}")
                    return None
            
            return self._resampler

        
        def _process_vad(self, pcm16_bytes: bytes) -> None:
            """Process audio through VAD (called from main thread, synchronous)."""
            if self._vad is None or VADState is None:
                return
            
            with self._vad_lock:
                pre_roll_max = int(pipecat_cfg.segment_sample_rate * pipecat_cfg.pre_roll_ms / 1000 * 2)
                self._pre_roll.extend(pcm16_bytes)
                if len(self._pre_roll) > pre_roll_max:
                    del self._pre_roll[:len(self._pre_roll) - pre_roll_max]
                
                try:
                    vad_state = self._vad.analyze_audio(pcm16_bytes)
                except Exception as e:
                    logger.debug(f"VAD analysis error: {e}")
                    return
                
                is_speech = vad_state in (VADState.STARTING, VADState.SPEAKING)
                was_speaking = self._in_speech
                
                # Barge-in: interrupt assistant when user starts speaking
                if is_speech and not was_speaking:
                    if self._out.is_bot_speaking():
                        self._out.interrupt()
                        logger.debug("Barge-in: interrupted assistant")
                
                if is_speech:
                    self._in_speech = True
                    self._silence_frames = 0
                    if not self._collecting:
                        self._collecting = True
                        self._utterance = bytearray(self._pre_roll)
                        self._interim_counter = 0
                    self._utterance.extend(pcm16_bytes)
                    
                    # Emit interim audio every ~500ms for live transcription
                    # (25 frames at 20ms each = 500ms)
                    self._interim_counter += 1
                    if self._interim_counter >= 25:
                        min_interim_bytes = int(pipecat_cfg.segment_sample_rate * 0.3 * 2)  # 300ms minimum
                        # Limit interim audio to ~25 seconds to avoid Whisper's 30s limit
                        max_interim_bytes = int(pipecat_cfg.segment_sample_rate * 25 * 2)  # 25 seconds max
                        if len(self._utterance) >= min_interim_bytes:
                            # Take only the last 25 seconds if utterance is too long
                            interim_data = bytes(self._utterance)
                            if len(interim_data) > max_interim_bytes:
                                interim_data = interim_data[-max_interim_bytes:]
                            # Put interim audio (limited size)
                            try:
                                self._interim_audio.put_nowait((interim_data, pipecat_cfg.segment_sample_rate))
                            except queue.Full:
                                pass  # Skip if queue is full
                        self._interim_counter = 0
                else:
                    if self._collecting:
                        self._utterance.extend(pcm16_bytes)
                        self._silence_frames += 1
                        
                        silence_threshold = int(pipecat_cfg.vad_stop_secs * 1000 / pipecat_cfg.frame_ms)
                        if self._silence_frames >= silence_threshold:
                            min_bytes = int(pipecat_cfg.segment_sample_rate * pipecat_cfg.min_utterance_ms / 1000 * 2)
                            if len(self._utterance) >= min_bytes:
                                self._segments.put((bytes(self._utterance), pipecat_cfg.segment_sample_rate))
                            self._collecting = False
                            self._utterance = bytearray()
                            self._pre_roll = bytearray()
                            self._in_speech = False
                            self._silence_frames = 0
                            self._interim_counter = 0
        
        @property
        def interim_audio(self) -> queue.Queue[Tuple[bytes, int]]:
            """Queue for interim audio chunks for live transcription."""
            return self._interim_audio
        
        @property
        def is_speaking(self) -> bool:
            """Check if user is currently speaking."""
            with self._vad_lock:
                return self._in_speech
        
        def recv(self, frame):
            """Process a single audio frame (synchronous callback)."""
            sample_rate = getattr(frame, "sample_rate", 48000) or 48000
            self._io_sample_rate = sample_rate
            
            resampler = self._get_resampler(frame)
            if resampler is not None:
                try:
                    for resampled_frame in resampler.resample(frame):
                        pcm = resampled_frame.to_ndarray()
                        if pcm.ndim == 2:
                            pcm = pcm[0]
                        if pcm.dtype != np.int16:
                            pcm = np.clip(pcm, -32768, 32767).astype(np.int16)
                        self._process_vad(pcm.tobytes())
                except ValueError as e:
                    logger.debug(f"Resampler mismatch, resetting: {e}")
                    self._resampler = None
                    self._last_frame_format = None
                except Exception as e:
                    logger.debug(f"Resampling error: {e}")
            
            frame_ms = pipecat_cfg.frame_ms
            samples_per_frame = int(sample_rate * frame_ms / 1000)
            bytes_per_frame = samples_per_frame * 2
            
            out_bytes = self._out.get_bytes_for_frame(bytes_per_frame)
            if not out_bytes or len(out_bytes) < bytes_per_frame:
                out_bytes = b"\x00" * bytes_per_frame
            
            try:
                audio_array = np.frombuffer(out_bytes[:bytes_per_frame], dtype=np.int16)
                if audio_array.size != samples_per_frame:
                    audio_array = np.zeros(samples_per_frame, dtype=np.int16)
                audio_array = audio_array.reshape(1, -1)
                out_frame = av.AudioFrame.from_ndarray(audio_array, format='s16', layout='mono')
                out_frame.sample_rate = sample_rate
                out_frame.pts = frame.pts
                return out_frame
            except Exception as e:
                logger.debug(f"Output frame error: {e}")
                return frame
        
        def __del__(self) -> None:
            self._stop.set()


    # ========================================================================
    # Audio Processor: webrtcvad fallback
    # ========================================================================
    
    class WebRTCVADProcessor(AudioProcessorBase):
        """Fallback audio processor using webrtcvad."""
        
        def __init__(self) -> None:
            self._segments: queue.Queue[Tuple[bytes, int]] = queue.Queue()
            self._byte_buffer = bytearray()
            self._current = bytearray()
            self._in_speech = False
            self._silence = 0
            self._speech_frames = 0
            self._vad = None
            self._sample_rate: Optional[int] = None
        
        @property
        def segments(self) -> queue.Queue[Tuple[bytes, int]]:
            return self._segments
        
        @property
        def output_sample_rate(self) -> int:
            return int(self._sample_rate or 48000)
        
        def _ensure_vad(self) -> None:
            if self._vad is None and _WEBRTCVAD_AVAILABLE and webrtcvad is not None:
                self._vad = webrtcvad.Vad(vad_cfg.aggressiveness)
        
        def recv(self, frame):
            self._ensure_vad()
            
            sample_rate = getattr(frame, "sample_rate", 48000) or 48000
            self._sample_rate = sample_rate
            frame_ms = vad_cfg.frame_ms
            samples_per_frame = int(sample_rate * frame_ms / 1000)
            bytes_per_frame = samples_per_frame * 2
            
            pcm = frame.to_ndarray()
            if pcm.ndim == 2:
                if pcm.shape[0] in (1, 2):
                    mono = pcm[0]
                else:
                    mono = pcm[:, 0]
            else:
                mono = pcm
            
            if mono.dtype != np.int16:
                mono = np.clip(mono, -32768, 32767).astype(np.int16)
            
            self._byte_buffer.extend(mono.tobytes())
            
            while len(self._byte_buffer) >= bytes_per_frame:
                chunk = bytes(self._byte_buffer[:bytes_per_frame])
                del self._byte_buffer[:bytes_per_frame]
                
                is_speech = False
                try:
                    if self._vad is not None:
                        is_speech = self._vad.is_speech(chunk, sample_rate)
                except Exception:
                    pass
                
                if is_speech:
                    self._in_speech = True
                    self._silence = 0
                    self._speech_frames += 1
                    self._current.extend(chunk)
                else:
                    if self._in_speech:
                        self._silence += 1
                        self._current.extend(chunk)
                        
                        if self._silence >= vad_cfg.silence_frames_to_finalize:
                            if self._speech_frames >= vad_cfg.min_speech_frames:
                                self._segments.put((bytes(self._current), sample_rate))
                            self._current = bytearray()
                            self._in_speech = False
                            self._silence = 0
                            self._speech_frames = 0
            
            return frame


    # ========================================================================
    # WebRTC Widget Setup
    # ========================================================================
    
    use_pipecat = _PIPECAT_AVAILABLE
    ProcessorClass = PipecatVADProcessor if use_pipecat else WebRTCVADProcessor
    webrtc_mode = WebRtcMode.SENDRECV if use_pipecat else WebRtcMode.SENDONLY
    
    # Session state key to track if greeting has been played
    greeting_played_key = f"realtime_greeting_played_{page_key}"
    
    # Pre-cache greeting audio on page load (runs in background)
    cache_key = f"{GREETING_CACHE_KEY}_{page_key}"
    if cache_key not in st.session_state:
        # Start background thread to cache greeting audio
        def cache_greeting_async():
            _get_or_cache_greeting_audio(page_key)
        
        threading.Thread(target=cache_greeting_async, daemon=True).start()
    
    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        webrtc_ctx = webrtc_streamer(
            key=f"realtime_voice_webrtc_{page_key}",
            mode=webrtc_mode,
            audio_processor_factory=ProcessorClass,
            media_stream_constraints={
                "audio": {
                    "echoCancellation": True,
                    "noiseSuppression": True,
                    "autoGainControl": True,
                    "sampleRate": 48000,
                },
                "video": False
            },
            async_processing=False,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        )
        
        with st.expander("Audio Settings", expanded=False):
            backend_info = "Pipecat Silero VAD" if use_pipecat else "webrtcvad"
            st.info(f"Using: {backend_info}")
            st.caption("Audio output is handled by your browser.")
    
    # ========================================================================
    # Play Greeting When Session Starts
    # ========================================================================
    
    # Check if WebRTC is playing and greeting hasn't been played yet
    if webrtc_ctx and getattr(webrtc_ctx.state, "playing", False):
        if not st.session_state.get(greeting_played_key, False):
            # Mark greeting as played (do this first to prevent re-triggering)
            st.session_state[greeting_played_key] = True
            
            # Get the appropriate messages key and greeting for this page
            messages_key = _get_messages_key(page_key)
            greeting_text = _get_realtime_greeting(page_key)
            
            # Add greeting to appropriate messages list
            st.session_state.setdefault(messages_key, [])
            st.session_state[messages_key].append({
                "role": "assistant", 
                "content": greeting_text
            })
            
            # Get cached or synthesize greeting audio
            greeting_audio = _get_or_cache_greeting_audio(page_key)
            
            if greeting_audio:
                # Try to play through WebRTC first (lower latency)
                greeting_played = False
                if webrtc_ctx.audio_processor and use_pipecat:
                    if hasattr(webrtc_ctx.audio_processor, "enqueue_assistant_pcm16"):
                        target_sr = webrtc_ctx.audio_processor.output_sample_rate
                        greeting_played = _play_greeting_via_webrtc(
                            webrtc_ctx.audio_processor, 
                            greeting_audio, 
                            target_sr
                        )
                
                # Fallback to audio widget if WebRTC playback failed
                if not greeting_played:
                    _play_greeting_via_audio_widget(greeting_audio)
            else:
                # Last resort: synthesize and play directly
                logger.warning("No cached greeting, synthesizing inline...")
                _synthesize_and_play(greeting_text)
            
            logger.info(f"AI greeting played on realtime voice session start for {page_key}")
    
    # Reset greeting flag when WebRTC stops
    if webrtc_ctx and not getattr(webrtc_ctx.state, "playing", False):
        if st.session_state.get(greeting_played_key, False):
            st.session_state[greeting_played_key] = False

    # ========================================================================
    # Process Finalized Segments with Streaming TTS
    # ========================================================================
    
    # Use page-specific messages key
    messages_key = _get_messages_key(page_key)
    if messages_key not in st.session_state:
        st.session_state[messages_key] = []
    
    # Session state for live transcription
    live_transcript_key = f"live_transcript_{page_key}"
    if live_transcript_key not in st.session_state:
        st.session_state[live_transcript_key] = ""
    
    # Placeholder for live transcription display - always visible
    live_transcript_placeholder = st.empty()
    
    # Display current live transcription state
    current_live_text = st.session_state.get(live_transcript_key, "")
    if current_live_text:
        with live_transcript_placeholder.container():
            st.markdown(
                f"""<div style="
                    padding: 0.75rem 1rem;
                    background: linear-gradient(135deg, #f0f4ff 0%, #e8f0fe 100%);
                    border-left: 4px solid #667eea;
                    border-radius: 8px;
                    margin-bottom: 1rem;
                    font-style: italic;
                    color: #555;
                    animation: pulse-subtle 1.5s infinite;
                ">
                    🎙️ <strong>Listening:</strong> {current_live_text}
                </div>
                <style>
                    @keyframes pulse-subtle {{
                        0%, 100% {{ opacity: 1; }}
                        50% {{ opacity: 0.8; }}
                    }}
                </style>""",
                unsafe_allow_html=True
            )

    if webrtc_ctx and webrtc_ctx.audio_processor:
        # ====================================================================
        # Live Transcription: Show partial text as user speaks
        # ====================================================================
        if use_pipecat and hasattr(webrtc_ctx.audio_processor, "interim_audio"):
            try:
                interim_pcm16, interim_sr = webrtc_ctx.audio_processor.interim_audio.get_nowait()
                if interim_pcm16 and interim_sr:
                    # Transcribe interim audio in a synchronous manner for immediate UI update
                    try:
                        wav_bytes = _pcm16_to_wav_bytes(interim_pcm16, sample_rate=interim_sr, channels=1)
                        speech_service = get_speech_service()
                        partial_text = speech_service.transcribe_audio(wav_bytes, sample_rate=16000)
                        if partial_text and partial_text.strip():
                            st.session_state[live_transcript_key] = partial_text.strip()
                            # Force UI update for live transcript
                            with live_transcript_placeholder.container():
                                st.markdown(
                                    f"""<div style="
                                        padding: 0.75rem 1rem;
                                        background: linear-gradient(135deg, #f0f4ff 0%, #e8f0fe 100%);
                                        border-left: 4px solid #667eea;
                                        border-radius: 8px;
                                        margin-bottom: 1rem;
                                        font-style: italic;
                                        color: #555;
                                    ">
                                        🎙️ <strong>Listening:</strong> {partial_text.strip()}
                                    </div>""",
                                    unsafe_allow_html=True
                                )
                    except Exception as e:
                        logger.debug(f"Interim transcription error: {e}")
            except queue.Empty:
                pass
        
        # ====================================================================
        # Process Completed Segments
        # ====================================================================
        try:
            pcm16, sr = webrtc_ctx.audio_processor.segments.get_nowait()
        except queue.Empty:
            pcm16, sr = None, None

        if pcm16 and sr:
            # Clear live transcript when we have a complete segment
            st.session_state[live_transcript_key] = ""
            live_transcript_placeholder.empty()
            
            # BARGE-IN: Interrupt any ongoing TTS playback immediately
            # 1. Clear the audio output queue to stop playback instantly
            if hasattr(webrtc_ctx.audio_processor, "interrupt_assistant"):
                webrtc_ctx.audio_processor.interrupt_assistant()
                logger.debug("Cleared audio output queue on barge-in")
            
            # 2. Interrupt the TTS handler to stop generating more audio
            active_tts_key = f"active_tts_handler_{page_key}"
            if active_tts_key in st.session_state and st.session_state[active_tts_key]:
                try:
                    st.session_state[active_tts_key].interrupt()
                    logger.debug("Interrupted TTS handler on barge-in")
                except Exception as e:
                    logger.debug(f"TTS interrupt error: {e}")
                st.session_state[active_tts_key] = None
            
            with st.spinner("Transcribing..."):
                wav_bytes = _pcm16_to_wav_bytes(pcm16, sample_rate=sr, channels=1)
                speech_service = get_speech_service()
                user_text = speech_service.transcribe_audio(wav_bytes, sample_rate=16000)

            if user_text and user_text.strip():
                user_text = user_text.strip()
                st.session_state[messages_key].append({"role": "user", "content": user_text})
                
                with st.chat_message("user"):
                    st.markdown(user_text)

                cfg = _get_chat_config_from_session(page_key)

                # Generate response with streaming TTS
                with st.chat_message("assistant"):
                    out = st.empty()
                    
                    # Set up streaming TTS handler for real-time synthesis
                    tts_handler = None
                    audio_processor = webrtc_ctx.audio_processor
                    tts_audio_queue = None
                    
                    if use_pipecat and hasattr(audio_processor, "enqueue_assistant_pcm16"):
                        # Create streaming TTS handler that synthesizes audio in parallel
                        target_sr = audio_processor.output_sample_rate
                        tts_audio_queue = queue.Queue()
                        tts_handler = StreamingTTSHandler(tts_audio_queue, target_sr)
                        
                        # Store in session state for barge-in interruption
                        st.session_state[active_tts_key] = tts_handler
                        
                        # Set up barge-in callback so VAD interrupt triggers TTS stop
                        if hasattr(audio_processor, "set_tts_interrupt_callback"):
                            audio_processor.set_tts_interrupt_callback(tts_handler.interrupt)
                        
                        # Reset interrupt flag for this new response
                        if hasattr(audio_processor, "reset_interrupt"):
                            audio_processor.reset_interrupt()
                        
                        # Create a callback that feeds text to TTS and queues audio
                        def tts_streaming_callback(text_chunk: str) -> None:
                            """Called for each LLM text chunk to synthesize audio in parallel."""
                            if tts_handler and not tts_handler.is_interrupted():
                                tts_handler.feed(text_chunk)
                                # Immediately drain any ready audio from queue to WebRTC
                                while True:
                                    try:
                                        audio_chunk = tts_audio_queue.get_nowait()
                                        if audio_chunk and not tts_handler.is_interrupted():
                                            audio_processor.enqueue_assistant_pcm16(audio_chunk)
                                    except queue.Empty:
                                        break
                    else:
                        tts_streaming_callback = None
                    
                    # Stream the response based on page type
                    if page_key == "rag":
                        # RAG query with document context
                        config = st.session_state.get("rag_config", {})
                        history = [{"role": msg["role"], "content": msg["content"]} 
                                  for msg in st.session_state[messages_key][:-1]]
                        
                        reply = streaming_handler.stream_rag_response(
                            query=user_text,
                            messages=history,
                            model=cfg["model"],
                            system_prompt=st.session_state.get("rag_system_prompt", ""),
                            n_results=config.get("n_results", 3),
                            use_multi_agent=config.get("use_multi_agent", False),
                            use_hybrid_search=config.get("use_hybrid_search", False),
                            backend=cfg["backend"],
                            hf_token=cfg.get("hf_token"),
                            placeholder=out,
                            tts_callback=tts_streaming_callback,
                        )
                    else:
                        # Regular chat response
                        reply = streaming_handler.stream_chat_response(
                            message=user_text,
                            model=cfg["model"],
                            temperature=cfg["temperature"],
                            backend=cfg["backend"],
                            hf_token=cfg.get("hf_token"),
                            placeholder=out,
                            use_fastapi=st.session_state.get("backend_available", False),
                            tts_callback=tts_streaming_callback,
                        )
                    
                    # Finalize any remaining TTS audio (if not interrupted)
                    if tts_handler:
                        if not tts_handler.is_interrupted():
                            tts_handler.finalize()
                            # Drain remaining audio to WebRTC
                            while tts_audio_queue:
                                try:
                                    audio_chunk = tts_audio_queue.get_nowait()
                                    if audio_chunk:
                                        audio_processor.enqueue_assistant_pcm16(audio_chunk)
                                except queue.Empty:
                                    break
                        tts_handler.shutdown()
                        st.session_state[active_tts_key] = None

                # Save response - prefer reply, but use partial text from TTS handler if interrupted
                final_reply = reply
                if tts_handler:
                    # If we have a TTS handler, get the full text it received
                    tts_full_text = tts_handler.get_full_text()
                    if tts_full_text and tts_full_text.strip():
                        # Use TTS handler's text if reply is empty or shorter (interrupted)
                        if not final_reply or len(tts_full_text) > len(final_reply or ""):
                            final_reply = tts_full_text
                
                if final_reply and final_reply.strip():
                    # Check if response was interrupted (TTS handler has interrupted flag)
                    was_interrupted = tts_handler and tts_handler.is_interrupted() if tts_handler else False
                    
                    # Add message with optional interrupted indicator
                    message_content = final_reply.strip()
                    if was_interrupted:
                        message_content = message_content + " [interrupted]"
                    
                    st.session_state[messages_key].append({"role": "assistant", "content": message_content})
                    
                    # If streaming TTS wasn't available, fall back to full synthesis
                    if not (use_pipecat and hasattr(audio_processor, "enqueue_assistant_pcm16")):
                        _synthesize_and_play(final_reply)
                else:
                    # Even if no reply, log error
                    logger.warning("No response received for realtime voice query")

                st.rerun()

    # Keep UI updating while call is active
    if webrtc_ctx and getattr(webrtc_ctx.state, "playing", False):
        time.sleep(0.2)
        st.rerun()
