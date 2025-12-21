"""Realtime voice chat component (WebRTC).

This provides a "speech → LLM → speech" loop inside the Streamlit UI.

Current scope:
- Captures microphone audio via WebRTC (streamlit-webrtc)
- Segments speech using Pipecat Silero VAD (preferred) or webrtcvad fallback
- Transcribes with the existing SpeechService (optionally faster-whisper backend)
- Streams an LLM reply using the existing streaming_handler (Ollama/vLLM)
- Synthesizes response audio via backend TTS endpoint (Kokoro/Edge-TTS)

Architecture:
- PipecatVADProcessor: Uses Pipecat Silero VAD for better speech detection
- WebRTCVADProcessor: Fallback using webrtcvad
- Both produce speech segments that are transcribed, processed by LLM, and synthesized to speech.
"""

from __future__ import annotations

import io
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

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
    silence_frames_to_finalize: int = 15  # ~300ms at 20ms frames
    min_speech_frames: int = 8  # ~160ms


@dataclass
class _PipecatConfig:
    """Configuration for Pipecat VAD."""
    frame_ms: int = 20
    segment_sample_rate: int = 16000
    pre_roll_ms: int = 200
    min_utterance_ms: int = 300
    # VAD parameters
    vad_start_secs: float = 0.2
    vad_stop_secs: float = 0.8
    vad_confidence: float = 0.7


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


def _get_chat_config_from_session() -> dict:
    """Extract chat configuration from session state."""
    backend_provider = st.session_state.get("current_backend", "ollama")
    temperature = st.session_state.get("chat_temperature", 0.7)
    if backend_provider == "vllm":
        model = st.session_state.get("vllm_model", "no-model")
        hf_token = st.session_state.get("hf_token")
    else:
        model = st.session_state.get("selected_chat_model", "default")
        hf_token = None

    return {
        "backend": backend_provider,
        "model": model,
        "temperature": temperature,
        "hf_token": hf_token,
    }


def _decode_audio_bytes_to_pcm16_mono(
    audio_bytes: bytes, *, target_sample_rate: int
) -> Tuple[bytes, int]:
    """Decode common audio formats (mp3/wav) into PCM16 mono at target_sample_rate.

    Uses PyAV (already required by streamlit-webrtc). Returns (pcm16_bytes, sample_rate).
    """
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

    def enqueue(self, pcm16: bytes) -> None:
        """Add audio to the output queue."""
        if not pcm16:
            return
        with self._lock:
            self._bot_speaking = True
            self._q.put(pcm16)

    def interrupt(self) -> None:
        """Clear all queued audio (user started speaking)."""
        with self._lock:
            self._bot_speaking = False
            self._buf.clear()
            try:
                while True:
                    self._q.get_nowait()
            except queue.Empty:
                pass

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
            st.session_state[realtime_mode_key] = False
            st.rerun()

    # Visual Indicator
    st.markdown('<div class="voice-container">', unsafe_allow_html=True)
    st.markdown('<div class="voice-circle">🎙️</div>', unsafe_allow_html=True)
    st.markdown('<div class="voice-status">Realtime Voice Active</div>', unsafe_allow_html=True)
    st.markdown('<div class="voice-subtext">Speak naturally. I will listen and respond.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ========================================================================
    # Audio Processor: Pipecat VAD (preferred)
    # ========================================================================
    
    class PipecatVADProcessor(AudioProcessorBase):
        """Audio processor using Pipecat Silero VAD for speech detection.
        
        This processor:
        1. Receives audio frames from WebRTC
        2. Resamples to 16kHz mono for VAD processing
        3. Detects speech segments using Silero VAD
        4. Queues complete utterances for transcription
        5. Outputs assistant audio back to the browser
        """
        
        def __init__(self) -> None:
            self._segments: queue.Queue[Tuple[bytes, int]] = queue.Queue()
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
        
        def _get_resampler(self, frame) -> Optional[Any]:
            """Get or create a resampler for the current frame format."""
            from av.audio.resampler import AudioResampler
            
            # Check if we need to recreate the resampler
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
                    logger.debug(f"Created resampler for format: {frame_format}")
                except Exception as e:
                    logger.warning(f"Failed to create resampler: {e}")
                    return None
            
            return self._resampler
        
        def _process_vad(self, pcm16_bytes: bytes) -> None:
            """Process audio through VAD (called from main thread, synchronous)."""
            if self._vad is None or VADState is None:
                return
            
            with self._vad_lock:
                # Pre-roll buffer for capturing audio before speech starts
                pre_roll_max = int(pipecat_cfg.segment_sample_rate * pipecat_cfg.pre_roll_ms / 1000 * 2)
                self._pre_roll.extend(pcm16_bytes)
                if len(self._pre_roll) > pre_roll_max:
                    del self._pre_roll[:len(self._pre_roll) - pre_roll_max]
                
                # Run VAD - THIS IS SYNCHRONOUS, NOT ASYNC!
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
                        # Start collecting, include pre-roll
                        self._collecting = True
                        self._utterance = bytearray(self._pre_roll)
                    self._utterance.extend(pcm16_bytes)
                else:
                    if self._collecting:
                        self._utterance.extend(pcm16_bytes)
                        self._silence_frames += 1
                        
                        # Check for end of utterance
                        silence_threshold = int(pipecat_cfg.vad_stop_secs * 1000 / pipecat_cfg.frame_ms)
                        if self._silence_frames >= silence_threshold:
                            # Finalize utterance
                            min_bytes = int(pipecat_cfg.segment_sample_rate * pipecat_cfg.min_utterance_ms / 1000 * 2)
                            if len(self._utterance) >= min_bytes:
                                self._segments.put((bytes(self._utterance), pipecat_cfg.segment_sample_rate))
                                logger.debug(f"Utterance finalized: {len(self._utterance)} bytes")
                            self._collecting = False
                            self._utterance = bytearray()
                            self._pre_roll = bytearray()
                            self._in_speech = False
                            self._silence_frames = 0
        
        def recv(self, frame):
            """Process a single audio frame (synchronous callback)."""
            sample_rate = getattr(frame, "sample_rate", 48000) or 48000
            self._io_sample_rate = sample_rate
            
            # Resample to 16kHz mono for VAD
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
                    # Frame format mismatch - reset resampler
                    logger.debug(f"Resampler mismatch, resetting: {e}")
                    self._resampler = None
                    self._last_frame_format = None
                except Exception as e:
                    logger.debug(f"Resampling error: {e}")
            
            # Generate output frame (silence or assistant audio)
            frame_ms = pipecat_cfg.frame_ms
            samples_per_frame = int(sample_rate * frame_ms / 1000)
            bytes_per_frame = samples_per_frame * 2
            
            out_bytes = self._out.get_bytes_for_frame(bytes_per_frame)
            if not out_bytes or len(out_bytes) < bytes_per_frame:
                out_bytes = b"\x00" * bytes_per_frame
            
            # Create output frame
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
            """For compatibility with PipecatVADProcessor."""
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
    
    # Choose processor based on availability
    use_pipecat = _PIPECAT_AVAILABLE
    ProcessorClass = PipecatVADProcessor if use_pipecat else WebRTCVADProcessor
    
    # WebRTC mode: SENDRECV for Pipecat (audio output), SENDONLY for fallback
    webrtc_mode = WebRtcMode.SENDRECV if use_pipecat else WebRtcMode.SENDONLY
    
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
            async_processing=False,  # Use synchronous processing
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        )
        
        # Audio settings expander
        with st.expander("Audio Settings", expanded=False):
            backend_info = "Pipecat Silero VAD" if use_pipecat else "webrtcvad"
            st.info(f"Using: {backend_info}")
            st.caption("Audio output is handled by your browser. Ensure the correct speaker is selected.")

    # ========================================================================
    # Process Finalized Segments
    # ========================================================================
    
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    if webrtc_ctx and webrtc_ctx.audio_processor:
        # Process at most 1 segment per rerun
        try:
            pcm16, sr = webrtc_ctx.audio_processor.segments.get_nowait()
        except queue.Empty:
            pcm16, sr = None, None

        if pcm16 and sr:
            with st.spinner("Transcribing..."):
                wav_bytes = _pcm16_to_wav_bytes(pcm16, sample_rate=sr, channels=1)
                speech_service = get_speech_service()
                user_text = speech_service.transcribe_audio(wav_bytes, sample_rate=16000)

            if user_text and user_text.strip():
                user_text = user_text.strip()
                st.session_state.chat_messages.append({"role": "user", "content": user_text})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(user_text)

                cfg = _get_chat_config_from_session()

                # Generate response
                with st.chat_message("assistant"):
                    out = st.empty()
                    reply = streaming_handler.stream_chat_response(
                        message=user_text,
                        model=cfg["model"],
                        temperature=cfg["temperature"],
                        backend=cfg["backend"],
                        hf_token=cfg.get("hf_token"),
                        placeholder=out,
                        use_fastapi=st.session_state.get("backend_available", False),
                    )

                if reply:
                    st.session_state.chat_messages.append({"role": "assistant", "content": reply})

                    # TTS: stream back via WebRTC if Pipecat, else use st.audio
                    tts_success = False
                    if use_pipecat and hasattr(webrtc_ctx.audio_processor, "enqueue_assistant_pcm16"):
                        try:
                            resp = requests.post(
                                f"{FASTAPI_URL}/api/tts/synthesize",
                                json={"text": reply, "voice": None, "use_cache": False},
                                timeout=120,
                            )
                            if resp.status_code == 200 and resp.content:
                                target_sr = webrtc_ctx.audio_processor.output_sample_rate
                                pcm_out, _ = _decode_audio_bytes_to_pcm16_mono(
                                    resp.content, target_sample_rate=target_sr
                                )
                                if pcm_out:
                                    webrtc_ctx.audio_processor.enqueue_assistant_pcm16(pcm_out)
                                    tts_success = True
                                    logger.debug(f"Queued TTS audio: {len(pcm_out)} bytes")
                            else:
                                logger.warning(f"TTS synthesis failed: {resp.status_code}")
                        except Exception as e:
                            logger.error(f"TTS queueing error: {e}")
                    
                    if not tts_success:
                        _synthesize_and_play(reply)

                st.rerun()

    # Keep UI updating while call is active
    if webrtc_ctx and getattr(webrtc_ctx.state, "playing", False):
        time.sleep(0.2)
        st.rerun()
