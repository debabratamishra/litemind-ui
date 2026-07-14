"""Audio processors for realtime voice chat (WebRTC).

Defines :class:`PipecatVADProcessor` (preferred, uses Pipecat Silero VAD) and
:class:`WebRTCVADProcessor` (fallback, uses webrtcvad). The module-level
availability detection mirrors the original lazy import strategy so the UI can
degrade gracefully when optional dependencies are missing.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from typing import Any, Callable, Optional, Tuple

import numpy as np

from .audio_output import _InterruptibleAudioOut
from .config import PIPECAT_CONFIG, VAD_CONFIG

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Availability detection (guarded imports — graceful degradation)
# ----------------------------------------------------------------------------

try:
    import av
    from streamlit_webrtc import AudioProcessorBase, WebRtcMode, webrtc_streamer

    _WEBRTC_DEPS_AVAILABLE = True
except ImportError:
    av = None
    AudioProcessorBase = object  # type: ignore[assignment, misc]
    WebRtcMode = None
    webrtc_streamer = None
    _WEBRTC_DEPS_AVAILABLE = False

try:
    import webrtcvad

    _WEBRTCVAD_AVAILABLE = True
except ImportError:
    webrtcvad = None
    _WEBRTCVAD_AVAILABLE = False

try:
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams, VADState

    _PIPECAT_AVAILABLE = True
except ImportError:
    SileroVADAnalyzer = None
    VADParams = None
    VADState = None
    _PIPECAT_AVAILABLE = False


# Public aliases used by the render orchestration
PIPECAT_AVAILABLE = _PIPECAT_AVAILABLE
WEBRTCVAD_AVAILABLE = _WEBRTCVAD_AVAILABLE
WEBRTC_DEPS_AVAILABLE = _WEBRTC_DEPS_AVAILABLE


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

        # Dedicated event loop for running async VAD calls from sync recv()
        self._vad_loop = asyncio.new_event_loop()
        self._vad_loop_thread = threading.Thread(
            target=self._vad_loop.run_forever, daemon=True, name="vad-loop"
        )
        self._vad_loop_thread.start()

        # Initialize VAD
        self._init_vad()

    def _init_vad(self) -> None:
        """Initialize Pipecat Silero VAD."""
        if not _PIPECAT_AVAILABLE or SileroVADAnalyzer is None:
            return
        try:
            self._vad = SileroVADAnalyzer(
                params=VADParams(
                    confidence=PIPECAT_CONFIG.vad_confidence,
                    start_secs=PIPECAT_CONFIG.vad_start_secs,
                    stop_secs=PIPECAT_CONFIG.vad_stop_secs,
                    min_volume=PIPECAT_CONFIG.vad_min_volume,
                )
            )
            self._vad.set_sample_rate(PIPECAT_CONFIG.segment_sample_rate)
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
                    rate=PIPECAT_CONFIG.segment_sample_rate,
                )
                self._last_frame_format = frame_format
            except Exception as e:
                logger.warning(f"Failed to create resampler: {e}")
                return None

        return self._resampler

    def _process_vad(self, pcm16_bytes: bytes) -> None:
        """Process audio through VAD (called from sync recv() callback).

        Pipecat 1.4.0 made analyze_audio an async coroutine. We submit it
        to the dedicated VAD event loop and block for the result so that
        the synchronous recv() callback still works correctly.
        """
        if self._vad is None or VADState is None:
            return

        with self._vad_lock:
            pre_roll_max = int(PIPECAT_CONFIG.segment_sample_rate * PIPECAT_CONFIG.pre_roll_ms / 1000 * 2)
            self._pre_roll.extend(pcm16_bytes)
            if len(self._pre_roll) > pre_roll_max:
                del self._pre_roll[: len(self._pre_roll) - pre_roll_max]

            try:
                # analyze_audio is a coroutine in Pipecat >= 1.4; run it on
                # our dedicated loop and wait for the result (timeout=0.1s).
                coro = self._vad.analyze_audio(pcm16_bytes)
                if asyncio.iscoroutine(coro):
                    future = asyncio.run_coroutine_threadsafe(coro, self._vad_loop)
                    vad_state = future.result(timeout=0.1)
                else:
                    vad_state = coro  # synchronous fallback (older Pipecat)
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
                    min_interim_bytes = int(PIPECAT_CONFIG.segment_sample_rate * 0.3 * 2)  # 300ms minimum
                    # Limit interim audio to ~25 seconds to avoid Whisper's 30s limit
                    max_interim_bytes = int(PIPECAT_CONFIG.segment_sample_rate * 25 * 2)  # 25 seconds max
                    if len(self._utterance) >= min_interim_bytes:
                        # Take only the last 25 seconds if utterance is too long
                        interim_data = bytes(self._utterance)
                        if len(interim_data) > max_interim_bytes:
                            interim_data = interim_data[-max_interim_bytes:]
                        # Put interim audio (limited size)
                        try:
                            self._interim_audio.put_nowait((interim_data, PIPECAT_CONFIG.segment_sample_rate))
                        except queue.Full:
                            pass  # Skip if queue is full
                    self._interim_counter = 0
            else:
                if self._collecting:
                    self._utterance.extend(pcm16_bytes)
                    self._silence_frames += 1

                    silence_threshold = int(PIPECAT_CONFIG.vad_stop_secs * 1000 / PIPECAT_CONFIG.frame_ms)
                    if self._silence_frames >= silence_threshold:
                        min_bytes = int(PIPECAT_CONFIG.segment_sample_rate * PIPECAT_CONFIG.min_utterance_ms / 1000 * 2)
                        if len(self._utterance) >= min_bytes:
                            self._segments.put((bytes(self._utterance), PIPECAT_CONFIG.segment_sample_rate))
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
                        if pcm.shape[0] in (1, 2):
                            pcm = pcm[0]
                        else:
                            pcm = pcm[:, 0]
                    if pcm.dtype != np.int16:
                        pcm = np.clip(pcm, -32768, 32767).astype(np.int16)
                    self._process_vad(pcm.tobytes())
            except ValueError as e:
                logger.debug(f"Resampler mismatch, resetting: {e}")
                self._resampler = None
                self._last_frame_format = None
            except Exception as e:
                logger.debug(f"Resampling error: {e}")

        frame_ms = PIPECAT_CONFIG.frame_ms
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
            out_frame = av.AudioFrame.from_ndarray(audio_array, format="s16", layout="mono")
            out_frame.sample_rate = sample_rate
            out_frame.pts = frame.pts
            return out_frame
        except Exception as e:
            logger.debug(f"Output frame error: {e}")
            return frame

    def __del__(self) -> None:
        self._stop.set()
        try:
            self._vad_loop.call_soon_threadsafe(self._vad_loop.stop)
        except Exception:
            pass


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
            self._vad = webrtcvad.Vad(VAD_CONFIG.aggressiveness)

    def recv(self, frame):
        self._ensure_vad()

        sample_rate = getattr(frame, "sample_rate", 48000) or 48000
        self._sample_rate = sample_rate
        frame_ms = VAD_CONFIG.frame_ms
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

                    if self._silence >= VAD_CONFIG.silence_frames_to_finalize:
                        if self._speech_frames >= VAD_CONFIG.min_speech_frames:
                            self._segments.put((bytes(self._current), sample_rate))
                        self._current = bytearray()
                        self._in_speech = False
                        self._silence = 0
                        self._speech_frames = 0

        return frame
