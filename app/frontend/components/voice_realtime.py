"""Realtime voice chat component (WebRTC).

This provides a "speech → LLM → speech" loop inside the Streamlit UI.

Current scope:
- Captures microphone audio via WebRTC (streamlit-webrtc)
- Segments speech using webrtcvad
- Transcribes with the existing SpeechService (optionally faster-whisper backend)
- Streams an LLM reply using the existing streaming_handler (Ollama/vLLM)
- Synthesizes response audio via backend TTS endpoint

Notes:
- Audio is played back in the UI after synthesis (not streamed frame-by-frame yet).
"""

from __future__ import annotations

import io
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any
from typing import Optional

import numpy as np
import requests
import streamlit as st

from ..config import FASTAPI_URL
from ..components.streaming_handler import streaming_handler
from ...services.speech_service import get_speech_service

logger = logging.getLogger(__name__)


@dataclass
class _VadConfig:
    aggressiveness: int = 2
    frame_ms: int = 20
    silence_frames_to_finalize: int = 12  # ~240ms at 20ms frames
    min_speech_frames: int = 8  # ~160ms


@dataclass
class _PipecatConfig:
    frame_ms: int = 20
    segment_sample_rate: int = 16000
    pre_roll_ms: int = 200
    min_utterance_ms: int = 250


def _pcm16_to_wav_bytes(pcm16: bytes, sample_rate: int, channels: int = 1) -> bytes:
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16)
    return buf.getvalue()


def _get_chat_config_from_session() -> dict:
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
) -> tuple[bytes, int]:
    """Decode common audio formats (mp3/wav) into PCM16 mono at target_sample_rate.

    Uses PyAV (already required by streamlit-webrtc). Returns (pcm16_bytes, sample_rate).
    """

    try:
        import av  # type: ignore
        from av.audio.resampler import AudioResampler  # type: ignore
    except Exception as e:
        raise RuntimeError("PyAV (av) is required to decode TTS audio") from e

    if not audio_bytes:
        return b"", target_sample_rate

    try:
        container: Any = av.open(io.BytesIO(audio_bytes))
    except Exception as e:
        logger.warning(f"PyAV failed to open audio bytes: {e}")
        # Fallback: if it's raw PCM or WAV header issue, we might fail here.
        # But pyttsx3 usually returns valid WAV.
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
                # Frame might be planar/float/etc; resampler normalizes it.
                for out_frame in resampler.resample(frame):
                    pcm = out_frame.to_ndarray()
                    if pcm.ndim == 2:
                        # (channels, samples)
                        pcm = pcm[0]
                    if pcm.dtype != np.int16:
                        pcm = np.clip(pcm, -32768, 32767).astype(np.int16)
                    out.extend(pcm.tobytes())
    except Exception as e:
        logger.warning(f"PyAV decoding error: {e}")
        return b"", target_sample_rate

    return bytes(out), target_sample_rate


def _synthesize_and_play(text: str, voice: Optional[str] = None) -> None:
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


def render_realtime_voice_chat(page_key: str = "chat") -> None:
    """Render realtime voice chat UI.

    This function manages its own session-state and writes to the shared
    `st.session_state.chat_messages` history for continuity with the Chat page.
    """

    realtime_mode_key = f"realtime_voice_mode_{page_key}"

    try:
        from streamlit_webrtc import AudioProcessorBase, WebRtcMode, webrtc_streamer  # type: ignore
        import av  # type: ignore
        import webrtcvad  # type: ignore
    except Exception as e:
        st.warning(
            "Realtime voice chat requires additional packages. "
            "Install `streamlit-webrtc`, `av`, and `webrtcvad` in your environment."
        )
        st.code(str(e))
        return

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

    # --- Audio processor implementation ---
    vad_cfg = _VadConfig()
    pipecat_cfg = _PipecatConfig()

    # Pipecat imports are optional (fallback to webrtcvad if missing).
    try:
        from pipecat.audio.vad.silero import SileroVADAnalyzer  # type: ignore
        from pipecat.audio.vad.vad_analyzer import VADParams, VADState  # type: ignore

        try:
            from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState  # type: ignore
            from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams  # type: ignore
            from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import (  # type: ignore
                LocalSmartTurnAnalyzerV3,
            )

            _SMART_TURN_AVAILABLE = True
        except Exception:
            LocalSmartTurnAnalyzerV3 = None  # type: ignore[assignment]
            SmartTurnParams = None  # type: ignore[assignment]
            EndOfTurnState = None  # type: ignore[assignment]
            _SMART_TURN_AVAILABLE = False

        _PIPECAT_AVAILABLE = True
    except Exception:
        SileroVADAnalyzer = None  # type: ignore[assignment]
        VADParams = None  # type: ignore[assignment]
        VADState = None  # type: ignore[assignment]
        _PIPECAT_AVAILABLE = False
        _SMART_TURN_AVAILABLE = False

    class _InterruptibleAudioOut:
        def __init__(self) -> None:
            self._q: queue.Queue[bytes] = queue.Queue()
            self._buf = bytearray()
            self._lock = threading.Lock()
            self._bot_speaking = False

        def enqueue(self, pcm16: bytes) -> None:
            if not pcm16:
                return
            with self._lock:
                self._bot_speaking = True
                self._q.put(pcm16)

        def interrupt(self) -> None:
            with self._lock:
                self._bot_speaking = False
                self._buf.clear()
                try:
                    while True:
                        self._q.get_nowait()
                except queue.Empty:
                    pass

        def is_bot_speaking(self) -> bool:
            with self._lock:
                return self._bot_speaking and (len(self._buf) > 0 or not self._q.empty())

        def get_bytes_for_frame(self, nbytes: int) -> bytes:
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

    class SegmentingAudioProcessor(AudioProcessorBase):
        def __init__(self) -> None:
            self._segments: queue.Queue[tuple[bytes, int]] = queue.Queue()
            self._byte_buffer = bytearray()
            self._current = bytearray()
            self._in_speech = False
            self._silence = 0
            self._speech_frames = 0

            self._vad: Optional[webrtcvad.Vad] = None
            self._sample_rate: Optional[int] = None

        @property
        def segments(self) -> queue.Queue[tuple[bytes, int]]:
            return self._segments

        def _ensure_vad(self) -> None:
            if self._vad is None:
                self._vad = webrtcvad.Vad(vad_cfg.aggressiveness)

        def recv(self, frame):
            # Pass-through: we only analyze input audio.
            self._ensure_vad()

            sample_rate = int(getattr(frame, "sample_rate", 48000) or 48000)
            self._sample_rate = sample_rate
            frame_ms = vad_cfg.frame_ms
            samples_per_frame = int(sample_rate * frame_ms / 1000)
            bytes_per_frame = samples_per_frame * 2  # mono int16

            pcm = frame.to_ndarray()
            if pcm.ndim == 2:
                # Handle both (channels, samples) and (samples, channels)
                if pcm.shape[0] in (1, 2):
                    mono = pcm[0]
                else:
                    mono = pcm[:, 0]
            else:
                mono = pcm

            if mono.dtype != np.int16:
                # best-effort conversion
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
                    # If frame sizes / rates mismatch, avoid crashing the stream
                    is_speech = False

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

    class PipecatBargeInAudioProcessor(AudioProcessorBase):
        """AudioProcessor that does:

        - Pipecat Silero VAD segmentation (optionally SmartTurn v3 end-of-turn)
        - True barge-in: when user starts speaking, clears queued output audio
        - Streams assistant audio back to the browser via WebRTC (SENDRECV)

        Notes:
        - This keeps the existing STT/LLM/TTS stack in Streamlit, but upgrades
          the realtime audio UX (turn-taking + interruption).
        """

        def __init__(self) -> None:
            self._segments: queue.Queue[tuple[bytes, int]] = queue.Queue()
            self._in_q: queue.Queue[bytes] = queue.Queue(maxsize=200)  # Balanced buffer size
            self._stop = threading.Event()

            self._out = _InterruptibleAudioOut()

            # WebRTC I/O sample rates. Input is typically 48k.
            self._io_sample_rate: Optional[int] = None

            # Resampler to 16k mono for VAD/turn.
            self._to_16k_resampler: Optional[Any] = None

            # Start worker thread lazily (after first audio frame, when we know IO SR).
            self._worker_thread: Optional[threading.Thread] = None

        @property
        def segments(self) -> queue.Queue[tuple[bytes, int]]:
            return self._segments

        @property
        def output_sample_rate(self) -> int:
            # Default to 48000 until we see a frame.
            return int(self._io_sample_rate or 48000)

        def enqueue_assistant_pcm16(self, pcm16: bytes) -> None:
            self._out.enqueue(pcm16)

        def interrupt_assistant(self) -> None:
            self._out.interrupt()

        def _ensure_worker(self) -> None:
            if self._worker_thread is not None:
                return
            self._worker_thread = threading.Thread(
                target=self._worker_main,
                name="pipecat-vad-worker",
                daemon=True,
            )
            self._worker_thread.start()

        def _worker_main(self) -> None:
            try:
                import asyncio

                asyncio.run(self._worker_loop())
            except Exception as e:
                logger.warning("Pipecat VAD worker stopped: %s", e)

        async def _worker_loop(self) -> None:
            import asyncio

            if not _PIPECAT_AVAILABLE or SileroVADAnalyzer is None or VADParams is None or VADState is None:
                return

            vad = SileroVADAnalyzer(
                params=VADParams(
                    # Increased stop_secs to prevent cutting off user mid-sentence
                    stop_secs=0.8,
                    start_secs=0.2,
                )
            )
            vad.set_sample_rate(pipecat_cfg.segment_sample_rate)

            turn = None
            if _SMART_TURN_AVAILABLE and LocalSmartTurnAnalyzerV3 is not None:
                try:
                    turn = LocalSmartTurnAnalyzerV3(params=SmartTurnParams())  # type: ignore[misc]
                    turn.set_sample_rate(pipecat_cfg.segment_sample_rate)
                except Exception:
                    turn = None

            # pre-roll ring buffer (bytes)
            pre_roll_max_bytes = int(
                pipecat_cfg.segment_sample_rate * (pipecat_cfg.pre_roll_ms / 1000.0) * 2
            )
            pre_roll = bytearray()

            vad_state = VADState.QUIET
            prev_vad_state = vad_state

            collecting = False
            utterance = bytearray()

            min_utt_bytes = int(
                pipecat_cfg.segment_sample_rate * (pipecat_cfg.min_utterance_ms / 1000.0) * 2
            )

            loop = asyncio.get_running_loop()

            while not self._stop.is_set():
                try:
                    # Use recv_queued logic implicitly by pulling from queue
                    # Reduced timeout for more responsive processing
                    audio = await asyncio.wait_for(loop.run_in_executor(None, self._in_q.get), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                if not audio:
                    continue

                # Update pre-roll
                pre_roll.extend(audio)
                if len(pre_roll) > pre_roll_max_bytes:
                    del pre_roll[: len(pre_roll) - pre_roll_max_bytes]

                prev_vad_state = vad_state
                vad_state = await vad.analyze_audio(audio)

                is_speech = vad_state in (VADState.STARTING, VADState.SPEAKING)
                if turn is not None:
                    end_state = turn.append_audio(audio, is_speech)
                else:
                    end_state = None

                # Barge-in: interrupt assistant audio as soon as we confirm speaking.
                if vad_state == VADState.SPEAKING and prev_vad_state != VADState.SPEAKING:
                    if self._out.is_bot_speaking():
                        self._out.interrupt()

                if is_speech:
                    if not collecting:
                        collecting = True
                        utterance = bytearray(pre_roll)
                    utterance.extend(audio)
                else:
                    if collecting:
                        # keep a bit of trailing silence for STT stability
                        utterance.extend(audio)

                # End-of-turn: prefer SmartTurn if available, else fall back to VAD quiet transition.
                should_finalize = False
                if turn is not None and EndOfTurnState is not None:
                    if end_state == EndOfTurnState.COMPLETE:
                        should_finalize = True
                    elif vad_state == VADState.QUIET and prev_vad_state != VADState.QUIET and turn.speech_triggered:
                        try:
                            state, _metrics = await turn.analyze_end_of_turn()
                            if state == EndOfTurnState.COMPLETE:
                                should_finalize = True
                        except Exception:
                            # If SmartTurn fails, fall back to VAD stop.
                            should_finalize = True
                else:
                    if vad_state == VADState.QUIET and prev_vad_state != VADState.QUIET:
                        should_finalize = True

                if should_finalize and collecting:
                    collecting = False
                    if len(utterance) >= min_utt_bytes:
                        self._segments.put((bytes(utterance), pipecat_cfg.segment_sample_rate))
                    utterance = bytearray()
                    pre_roll = bytearray()
                    if turn is not None:
                        try:
                            turn.clear()
                        except Exception:
                            pass

        async def recv(self, frame):
            """Legacy recv method for single frame processing.
            
            Note: streamlit-webrtc calls either recv() or recv_queued().
            Since we implemented recv_queued(), this might not be called,
            but we keep it for compatibility.
            """
            result = await self.recv_queued([frame])
            return result[0] if result else frame

        async def recv_queued(self, frames):
            """Process queued frames to avoid dropping audio (async version)."""
            self._ensure_worker()
            
            if not frames:
                return []

            # Use the first frame to determine sample rate
            first_frame = frames[0]
            sample_rate = int(getattr(first_frame, "sample_rate", 48000) or 48000)
            self._io_sample_rate = sample_rate

            # Lazily build a resampler for input -> 16k mono s16.
            if self._to_16k_resampler is None:
                from av.audio.resampler import AudioResampler  # type: ignore

                self._to_16k_resampler = AudioResampler(
                    format="s16",
                    layout="mono",
                    rate=pipecat_cfg.segment_sample_rate,
                )

            # Resample incoming audio to 16k mono PCM16 for VAD/turn.
            try:
                resampler = self._to_16k_resampler
                if resampler is None:
                    raise RuntimeError("Resampler not initialized")

                for frame in frames:
                    # PyAV < 10.0.0 resample() returns a generator or list of frames
                    # Validate frame before resampling to prevent AudioResampler errors
                    try:
                        # Validate frame has required attributes
                        if not hasattr(frame, 'format') or not hasattr(frame, 'layout'):
                            logger.debug("Frame missing required attributes, skipping")
                            continue
                        
                        # Validate frame format matches resampler expectations
                        if frame.samples == 0:
                            logger.debug("Frame has no samples, skipping")
                            continue
                        
                        resampled_frames = resampler.resample(frame)
                        for f in resampled_frames:
                            pcm = f.to_ndarray()
                            if pcm.ndim == 2:
                                pcm = pcm[0]
                            if pcm.dtype != np.int16:
                                pcm = np.clip(pcm, -32768, 32767).astype(np.int16)
                            try:
                                # Use put with timeout instead of put_nowait to handle backpressure
                                self._in_q.put(pcm.tobytes(), timeout=0.01)
                            except queue.Full:
                                # Drop frames if queue is full to prevent blocking
                                logger.debug("Input queue full, dropping frame")
                                pass
                    except ValueError as e:
                        # Catch specific AudioResampler setup mismatch errors
                        logger.debug(f"Frame format mismatch for resampler: {e}")
                        continue
                    except Exception as e:
                        logger.debug(f"Error resampling frame: {e}")
                        continue
            except Exception as e:
                logger.debug(f"Error in audio resampling: {e}")
                pass

            # Output audio at the IO sample rate in 20ms frames.
            import av  # type: ignore

            frame_ms = pipecat_cfg.frame_ms
            samples_per_frame = int(sample_rate * frame_ms / 1000)
            bytes_per_frame = samples_per_frame * 2
            
            output_frames = []
            # Only process the number of frames we actually received to avoid queue buildup
            num_frames = min(len(frames), 5)  # Limit to 5 frames per batch to reduce lag
            
            for _ in range(num_frames):
                out_bytes = self._out.get_bytes_for_frame(bytes_per_frame)
                if not out_bytes or len(out_bytes) < bytes_per_frame:
                    out_bytes = b"\x00" * bytes_per_frame

                # Create frame with proper initialization using from_ndarray
                try:
                    # Validate output bytes length
                    if len(out_bytes) != bytes_per_frame:
                        logger.debug(f"Output bytes length mismatch: expected {bytes_per_frame}, got {len(out_bytes)}")
                        out_bytes = out_bytes[:bytes_per_frame].ljust(bytes_per_frame, b"\x00")
                    
                    # Convert bytes to numpy array with correct shape
                    audio_array = np.frombuffer(out_bytes, dtype=np.int16)
                    
                    # Validate array size
                    if audio_array.size != samples_per_frame:
                        logger.debug(f"Audio array size mismatch: expected {samples_per_frame}, got {audio_array.size}")
                        audio_array = np.pad(audio_array, (0, max(0, samples_per_frame - audio_array.size)), mode='constant')[:samples_per_frame]
                    
                    # Reshape to (1, samples) for mono audio
                    audio_array = audio_array.reshape(1, -1)
                    
                    # Create frame from numpy array
                    out_frame = av.AudioFrame.from_ndarray(audio_array, format='s16', layout='mono')
                    out_frame.sample_rate = sample_rate
                    out_frame.pts = None
                    
                    output_frames.append(out_frame)
                except Exception as e:
                    logger.debug(f"Error creating output frame: {e}")
                    # Return silence frame on error using same method
                    try:
                        silence_array = np.zeros((1, samples_per_frame), dtype=np.int16)
                        silence_frame = av.AudioFrame.from_ndarray(silence_array, format='s16', layout='mono')
                        silence_frame.sample_rate = sample_rate
                        silence_frame.pts = None
                        output_frames.append(silence_frame)
                    except Exception:
                        pass
                
            return output_frames

        def __del__(self) -> None:
            try:
                self._stop.set()
            except Exception:
                pass

    # --- WebRTC widget ---
    use_pipecat = _PIPECAT_AVAILABLE
    audio_processor_factory = PipecatBargeInAudioProcessor if use_pipecat else SegmentingAudioProcessor

    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        webrtc_ctx = webrtc_streamer(
            key=f"realtime_voice_webrtc_{page_key}",
            mode=WebRtcMode.SENDRECV if use_pipecat else WebRtcMode.SENDONLY,
            audio_processor_factory=audio_processor_factory,
            media_stream_constraints={
                "audio": {
                    "echoCancellation": True,
                    "noiseSuppression": True,
                    "autoGainControl": True,
                    "sampleRate": 48000,
                },
                "video": False
            },
            async_processing=True,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        )
        
        # Audio device selection (hidden by default, but useful for debugging)
        with st.expander("Audio Settings", expanded=False):
            st.info("If you can't hear audio, check your system output device.")
            # Note: Browser handles device selection for WebRTC, but we can show a hint.
            st.caption("Audio output is handled by your browser. Ensure the correct speaker is selected in your OS or Browser settings.")

    # --- Process finalized segments ---
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    if webrtc_ctx and webrtc_ctx.audio_processor:
        processed_any = False

        # Process at most 1 segment per rerun to keep UI responsive.
        try:
            pcm16, sr = webrtc_ctx.audio_processor.segments.get_nowait()
            processed_any = True
        except queue.Empty:
            pcm16, sr = None, None

        if processed_any and pcm16 and sr:
            with st.spinner("Transcribing..."):
                wav_bytes = _pcm16_to_wav_bytes(pcm16, sample_rate=sr, channels=1)
                speech_service = get_speech_service()
                user_text = speech_service.transcribe_audio(wav_bytes, sample_rate=16000)

            if user_text and user_text.strip():
                user_text = user_text.strip()
                st.session_state.chat_messages.append({"role": "user", "content": user_text})

                cfg = _get_chat_config_from_session()

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

                    # If we are in Pipecat+barge-in mode, stream assistant audio back over WebRTC
                    # so it can be interrupted. Otherwise keep the old st.audio path.
                    tts_success = False
                    if use_pipecat and hasattr(webrtc_ctx.audio_processor, "enqueue_assistant_pcm16"):
                        try:
                            resp = requests.post(
                                f"{FASTAPI_URL}/api/tts/synthesize",
                                json={"text": reply, "voice": None, "use_cache": False},
                                timeout=120,
                            )
                            if resp.status_code == 200 and resp.content:
                                target_sr = int(getattr(webrtc_ctx.audio_processor, "output_sample_rate", 48000))
                                pcm_out, _ = _decode_audio_bytes_to_pcm16_mono(
                                    resp.content, target_sample_rate=target_sr
                                )
                                webrtc_ctx.audio_processor.enqueue_assistant_pcm16(pcm_out)  # type: ignore[attr-defined]
                                tts_success = True
                            else:
                                logger.warning(
                                    "TTS synthesis failed: %s %s", resp.status_code, resp.text
                                )
                                st.error(f"TTS Error: {resp.status_code} - {resp.text}")
                        except Exception as e:
                            logger.error("TTS queueing error: %s", e)
                            st.error(f"TTS Queueing Error: {e}")
                    
                    if not tts_success:
                        _synthesize_and_play(reply)

                # Nudge Streamlit to refresh chat history area.
                st.rerun()

    # If call is active, keep UI updating to pick up segments.
    if webrtc_ctx and getattr(webrtc_ctx.state, "playing", False):
        time.sleep(0.15)
        st.rerun()
