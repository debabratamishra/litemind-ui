"""Realtime voice pipeline: WebRTC (Pipecat) + sandwich STT -> LLM -> TTS.

STT  -> app.services.speech_service   (Whisper, transformers)
LLM  -> app.services.llm_gateway       (chosen provider via LiteLLM/Ollama)
TTS  -> app.services.tts_service       (Kokoro, primary)
VAD  -> Pipecat SileroVADAnalyzer
Events -> WebRTC data channel (SmallWebRTCConnection.send_app_message)
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import soundfile as sf
from openai import AsyncOpenAI
from pipecat.audio.vad.silero import SileroVADAnalyzer
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
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.openai.base_llm import BaseOpenAILLMService, OpenAILLMSettings
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.workers.runner import WorkerRunner

from app.services.llm_gateway import stream_completion
from app.services.speech_service import get_speech_service
from app.services.tts_service import get_tts_service

DEFAULT_SYSTEM_INSTRUCTION = (
    "You are a helpful voice assistant. Respond briefly and conversationally. "
    "Avoid emojis, bullet points, and any formatting that cannot be spoken aloud."
)


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


class BackendWhisperSTTService(STTService):
    """Delegates to the existing Whisper speech_service."""

    async def run_stt(self, audio: bytes):
        text = get_speech_service().transcribe_audio(audio, sample_rate=16000)
        if text:
            yield TranscriptionFrame(
                text=text, user_id="user", timestamp="", finalized=True
            )


class BackendKokoroTTSService(TTSService):
    """Delegates to the existing Kokoro tts_service (returns WAV -> 16-bit PCM)."""

    def __init__(self, voice: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self._voice = voice

    async def run_tts(self, text: str, context_id: str):
        wav = get_tts_service().synthesize_text_chunk(text, self._voice)
        if not wav:
            return
        audio_array, _ = sf.read(io.BytesIO(wav), dtype="int16", always_2d=False)
        pcm = np.asarray(audio_array, dtype=np.int16).tobytes()
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


def build_voice_pipeline(connection: SmallWebRTCConnection, settings: VoiceSettings):
    transport = SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    )
    stt = BackendWhisperSTTService()
    tts = BackendKokoroTTSService(voice=settings.voice)
    llm = BackendLLMService(settings)

    context = LLMContext()
    context.add_message({"role": "system", "content": settings.system_instruction})

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
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
