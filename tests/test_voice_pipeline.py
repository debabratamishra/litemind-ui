import io
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import soundfile as sf
from pipecat.frames.frames import (
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.stt_service import SegmentedSTTService

from app.services.voice_pipeline import (
    AssistantTranscriptEmitter,
    BackendKokoroTTSService,
    BackendLLMService,
    BackendWhisperSTTService,
    UserTranscriptEmitter,
    VoiceSettings,
    build_voice_pipeline,
)


def _fake_wav():
    buf = io.BytesIO()
    sf.write(buf, np.zeros(16, dtype=np.float32), 24000, format="WAV")
    return buf.getvalue()


@pytest.fixture
def patch_services(monkeypatch):
    import app.services.voice_pipeline as vp

    class FakeSpeech:
        def transcribe_pcm(self, audio_data, source_rate=16000):
            return "hello world"

    class FakeTTS:
        def synthesize_text_chunk(self, text, voice=None):
            return _fake_wav()

    monkeypatch.setattr(vp, "get_speech_service", lambda: FakeSpeech())
    monkeypatch.setattr(vp, "get_tts_service", lambda: FakeTTS())


async def test_stt_yields_transcription_frame(patch_services):
    svc = BackendWhisperSTTService()
    frames = [f async for f in svc.run_stt(b"fake-wav-bytes")]
    texts = [f for f in frames if isinstance(f, TranscriptionFrame)]
    assert texts, "expected a TranscriptionFrame"
    assert texts[0].text == "hello world"
    assert texts[0].finalized is True


def test_stt_is_segmented_not_per_frame():
    """Whisper is a batch model: STT must be SegmentedSTTService so it only
    transcribes on VAD speech-stop, not on every ~20ms frame (which floods
    Whisper with silent chunks and makes it hallucinate)."""
    svc = BackendWhisperSTTService()
    assert isinstance(svc, SegmentedSTTService)
    # Whisper consumes the raw PCM buffer directly, not a WAV container.
    assert svc.wants_wav_segments is False


async def test_tts_yields_pcm_audio_frame(patch_services):
    svc = BackendKokoroTTSService(voice="af_heart")
    frames = [f async for f in svc.run_tts("hello", "ctx-1")]
    audio = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
    assert audio, "expected a TTSAudioRawFrame"
    assert audio[0].sample_rate == 24000
    assert audio[0].num_channels == 1
    # OutputAudioRawFrame expects 16-bit PCM => 2 bytes per sample per channel
    assert len(audio[0].audio) % 2 == 0


async def test_llm_streams_text_frames(monkeypatch):
    import app.services.voice_pipeline as vp

    async def fake_stream(messages, **kwargs):
        for t in ["Hello", " world"]:
            yield t

    monkeypatch.setattr(vp, "stream_completion", fake_stream)
    svc = BackendLLMService(VoiceSettings(model="llama3", backend="ollama"))
    captured = []

    async def cap(frame, direction=None):
        captured.append(frame)

    svc.push_frame = cap
    ctx = LLMContext()
    ctx.add_message({"role": "user", "content": "hi"})
    await svc._process_context(ctx)
    joined = "".join(f.text for f in captured if isinstance(f, LLMTextFrame))
    assert joined == "Hello world"


async def test_llm_process_frame_drives_inference(monkeypatch):
    import app.services.voice_pipeline as vp

    async def fake_stream(messages, **kwargs):
        for t in ["Hello", " world"]:
            yield t

    monkeypatch.setattr(vp, "stream_completion", fake_stream)
    svc = BackendLLMService(VoiceSettings(model="llama3", backend="ollama"))
    captured = []

    async def cap(frame, direction=None):
        captured.append(frame)

    svc.push_frame = cap
    ctx = LLMContext(messages=[{"role": "user", "content": "hi"}])
    await svc.process_frame(
        LLMContextFrame(context=ctx), FrameDirection.DOWNSTREAM
    )

    assert any(
        isinstance(f, LLMFullResponseStartFrame) for f in captured
    ), "expected LLMFullResponseStartFrame around the stream"
    assert any(
        isinstance(f, LLMFullResponseEndFrame) for f in captured
    ), "expected LLMFullResponseEndFrame around the stream"
    text_frames = [f for f in captured if isinstance(f, LLMTextFrame)]
    assert text_frames, "expected at least one LLMTextFrame"
    assert "".join(f.text for f in text_frames) == "Hello world"


def test_build_voice_pipeline_constructs_without_real_peer(monkeypatch):
    """build_voice_pipeline must construct cheaply and network-free.

    SileroVADAnalyzer (downloads a torch model) and SmallWebRTCTransport
    (binds to a real WebRTC peer) are monkeypatched to dummies so we can
    assert the pipeline is built without loading heavy weights or touching
    a real connection.
    """
    import app.services.voice_pipeline as vp

    class FakeVAD:
        pass

    class FakeTransport(FrameProcessor):
        def __init__(self, webrtc_connection, params, **kwargs):
            super().__init__()
            self.webrtc_connection = webrtc_connection
            self.params = params

        def input(self):
            return FrameProcessor()

        def output(self):
            return FrameProcessor()

    monkeypatch.setattr(vp, "SileroVADAnalyzer", lambda *a, **k: FakeVAD())
    monkeypatch.setattr(vp, "SmallWebRTCTransport", FakeTransport)

    connection = MagicMock()
    pipeline, transport = build_voice_pipeline(connection, VoiceSettings())
    assert isinstance(pipeline, Pipeline)
    assert isinstance(transport, FakeTransport)
    # The connection is threaded through to the emitters / transport.
    assert transport.webrtc_connection is connection

    # Exactly one VAD source, placed upstream of the STT (so the segmented STT
    # flushes on speech stop) and not duplicated in the aggregator.
    vad_processors = [p for p in pipeline.processors if isinstance(p, VADProcessor)]
    assert len(vad_processors) == 1, "expected a single VADProcessor in the pipeline"


async def test_emitters_send_expected_event_json():
    """Emitters must push the documented event JSON over the data channel.

    UserTranscriptEmitter -> {"type":"user_transcript","text":...,"final":True}
    AssistantTranscriptEmitter -> {"type":"assistant_text","text":...}
                               -> {"type":"assistant_end"} on turn end.
    """
    sent = []
    fake_conn = MagicMock()
    fake_conn.send_app_message = lambda msg: sent.append(msg)

    user_emitter = UserTranscriptEmitter(fake_conn)
    # Emitters must pass frames downstream (FrameProcessor does not do this
    # automatically); mock push_frame so the direct call doesn't touch a pipeline.
    user_emitter.push_frame = AsyncMock()
    await user_emitter.process_frame(
        TranscriptionFrame(text="hello there", user_id="user", timestamp="", finalized=True),
        FrameDirection.DOWNSTREAM,
    )
    # The TranscriptionFrame must be forwarded downstream to the context aggregator.
    user_emitter.push_frame.assert_awaited_once()

    assistant_emitter = AssistantTranscriptEmitter(fake_conn)
    assistant_emitter.push_frame = AsyncMock()
    await assistant_emitter.process_frame(
        LLMTextFrame(text="Hi there"), FrameDirection.DOWNSTREAM
    )
    await assistant_emitter.process_frame(
        TextFrame(text=" how are you"), FrameDirection.DOWNSTREAM
    )
    await assistant_emitter.process_frame(
        LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM
    )
    # Every frame must be forwarded downstream to the TTS service / aggregator.
    assert assistant_emitter.push_frame.await_count == 3

    user_events = [m for m in sent if m.get("type") == "user_transcript"]
    assert user_events, "expected a user_transcript event over the data channel"
    assert user_events[0] == {"type": "user_transcript", "text": "hello there", "final": True}

    assistant_text_events = [m for m in sent if m.get("type") == "assistant_text"]
    assert assistant_text_events, "expected assistant_text events over the data channel"
    joined = "".join(m["text"] for m in assistant_text_events)
    assert joined == "Hi there how are you"

    end_events = [m for m in sent if m.get("type") == "assistant_end"]
    assert end_events == [{"type": "assistant_end"}]
