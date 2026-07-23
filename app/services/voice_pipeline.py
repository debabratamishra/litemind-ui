"""Realtime voice pipeline: WebRTC (Pipecat) + sandwich STT -> LLM -> TTS.

STT  -> app.services.speech_service   (Whisper, transformers)
LLM  -> app.services.llm_gateway       (chosen provider via LiteLLM/Ollama)
TTS  -> app.services.tts_service       (Kokoro, primary)
VAD  -> Pipecat SileroVADAnalyzer
Events -> WebRTC data channel (SmallWebRTCConnection.send_app_message)
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import soundfile as sf
from openai import AsyncOpenAI
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMTextFrame,
    TextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.openai.base_llm import BaseOpenAILLMService, OpenAILLMSettings
from pipecat.services.settings import STTSettings, TTSSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.services.tts_service import TTSService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.workers.runner import WorkerRunner

from app.services.llm_gateway import stream_completion
from app.services.speech_service import get_speech_service
from app.services.tts_service import get_tts_service

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_INSTRUCTION = (
    "You are a helpful voice assistant. Respond briefly and conversationally. "
    "Avoid emojis, bullet points, and any formatting that cannot be spoken aloud."
)

# Silence (seconds) required before VAD declares an utterance ended. Pipecat's
# default is 0.2s, which splits a single thought into multiple turns on any
# brief pause. 1.2s tolerates natural mid-sentence gaps so the LLM is invoked
# once per complete utterance.
VAD_STOP_SECS = 1.2

# Default Kokoro voice used when the client does not request one.
DEFAULT_TTS_VOICE = "af_heart"


@dataclass
class VoiceSettings:
    model: str | None = None
    backend: str | None = None
    api_key: str | None = None
    api_base: str | None = None
    temperature: float = 0.7
    max_tokens: int = 512
    voice: str | None = None
    system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION
    user_id: str | None = None  # Owning user (for session/transcript isolation)


class BackendWhisperSTTService(SegmentedSTTService):
    """Delegates to the existing Whisper speech_service.

    Whisper is a batch model, so we extend ``SegmentedSTTService`` which buffers
    audio and only runs STT when VAD reports the user stopped speaking. The base
    ``STTService`` fires on every ~20 ms frame, which floods Whisper with
    near-silent chunks and makes it hallucinate ("Thank you for watching").

    The buffered segment is raw 16-bit mono PCM, decoded via
    ``SpeechService.transcribe_pcm`` (not an encoded file).
    """

    def __init__(self, **kwargs):
        # model/language are not applicable to our backend STT — mark them
        # explicitly as None to satisfy Pipecat's STTSettings validation
        # (otherwise it warns "fields are NOT_GIVEN").
        super().__init__(
            settings=STTSettings(model=None, language=None),
            sample_rate=16000,
            **kwargs,
        )

    @property
    def wants_wav_segments(self) -> bool:
        # Whisper reads the raw PCM buffer directly; wrapping it in a WAV
        # container would make transcribe_pcm misinterpret the 44-byte header.
        return False

    async def run_stt(self, audio: bytes):
        text = get_speech_service().transcribe_pcm(audio, source_rate=self.sample_rate)
        if text:
            yield TranscriptionFrame(
                text=text, user_id="user", timestamp="", finalized=True
            )


class BackendKokoroTTSService(TTSService):
    """Delegates to the existing Kokoro tts_service (returns WAV -> 16-bit PCM)."""

    def __init__(self, voice: str | None = None, **kwargs):
        # Initialize TTSSettings fields explicitly (None for unsupported) to
        # avoid Pipecat's "fields are NOT_GIVEN" warning. Synthesis is driven by
        # our own Kokoro-backed get_tts_service(), not Pipecat's client.
        super().__init__(settings=TTSSettings(model=None, voice=None, language=None), **kwargs)
        self._voice = voice

    async def run_tts(self, text: str, context_id: str):
        logger.info("TTS run_tts: synthesizing %d chars (voice=%s)", len(text), self._voice)
        wav = get_tts_service().synthesize_text_chunk(text, self._voice)
        if not wav:
            logger.warning("TTS run_tts: Kokoro returned no audio for %r", text[:40])
            return
        audio_array, _ = sf.read(io.BytesIO(wav), dtype="int16", always_2d=False)
        pcm = np.asarray(audio_array, dtype=np.int16).tobytes()
        logger.info("TTS run_tts: produced %d bytes of PCM audio", len(pcm))
        yield TTSAudioRawFrame(
            audio=pcm, sample_rate=24000, num_channels=1, context_id=context_id
        )


class BackendLLMService(BaseOpenAILLMService):
    """Streams tokens via the existing llm_gateway (chosen backend)."""

    def __init__(self, settings: VoiceSettings, **kwargs):
        self._voice_settings = settings
        super().__init__(settings=OpenAILLMSettings(model=settings.model or ""))

    def create_client(
        self,
        api_key=None,
        base_url=None,
        organization=None,
        project=None,
        default_headers=None,
        **kwargs,
    ):
        # We never use an AsyncOpenAI client — inference is delegated to the
        # existing llm_gateway. Return None at runtime; we type it as the base
        # client so the override still satisfies LSP.
        return cast(AsyncOpenAI, None)

    async def _process_context(self, context: LLMContext):
        raw_messages = cast("list[dict[str, Any]]", context.get_messages())
        messages = [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in raw_messages
        ]
        async for delta in stream_completion(
            messages,
            backend=self._voice_settings.backend,
            model=self._voice_settings.model,
            api_key=self._voice_settings.api_key,
            api_base=self._voice_settings.api_base,
            temperature=self._voice_settings.temperature,
            max_tokens=self._voice_settings.max_tokens,
        ):
            if delta:
                await self.push_frame(LLMTextFrame(text=delta))


class UserTranscriptEmitter(FrameProcessor):
    """Sends finalized user transcripts to the browser over the data channel."""

    def __init__(self, connection: SmallWebRTCConnection):
        super().__init__()
        self._send = connection.send_app_message

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.finalized:
            self._send({"type": "user_transcript", "text": frame.text, "final": True})
        # Pass the frame downstream. FrameProcessor does NOT forward frames on
        # its own, so without this the TranscriptionFrame is swallowed here and
        # never reaches the user context aggregator — the LLM never sees the
        # user's turn and never runs.
        await self.push_frame(frame, direction)


class AssistantTranscriptEmitter(FrameProcessor):
    """Sends streamed assistant text + turn-end to the browser over the data channel."""

    def __init__(self, connection: SmallWebRTCConnection):
        super().__init__()
        self._send = connection.send_app_message

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, (TextFrame, LLMTextFrame)) and getattr(frame, "text", ""):
            self._send({"type": "assistant_text", "text": frame.text})
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._send({"type": "assistant_end"})
        # Pass the frame downstream. Without this the LLMTextFrame / end frame is
        # swallowed here and never reaches the TTS service or the assistant
        # context aggregator, so no audio is ever produced.
        await self.push_frame(frame, direction)


def build_voice_pipeline(connection: SmallWebRTCConnection, settings: VoiceSettings):
    transport = SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    )
    stt = BackendWhisperSTTService()
    tts = BackendKokoroTTSService(voice=settings.voice or DEFAULT_TTS_VOICE)
    llm = BackendLLMService(settings)

    context = LLMContext()
    context.add_message({"role": "system", "content": settings.system_instruction})

    # A single VAD source. It runs upstream of the STT so the SegmentedSTTService
    # flushes its buffer on speech stop, and its VAD frames also drive the
    # user-turn controller in the aggregator. (The aggregator therefore must NOT
    # run its own VAD analyzer, or we'd get double VAD frames.)
    # stop_secs raises the silence threshold before an utterance is considered
    # finished, so a brief mid-thought pause doesn't start a new user turn.
    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=VAD_STOP_SECS)))
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            vad,
            stt,
            UserTranscriptEmitter(connection),
            user_aggregator,
            llm,
            AssistantTranscriptEmitter(connection),
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )
    return pipeline, transport


async def run_voice_pipeline(connection: SmallWebRTCConnection, settings: VoiceSettings):
    pipeline, transport = build_voice_pipeline(connection, settings)
    worker = PipelineWorker(pipeline, params=PipelineParams(enable_metrics=False))
    runner = WorkerRunner(handle_sigint=False)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        connection.send_app_message({"type": "ready"})

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await worker.cancel()

    await runner.add_workers(worker)
    await runner.run()
