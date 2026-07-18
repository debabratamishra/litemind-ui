import asyncio
import io
import soundfile as sf
import numpy as np
import pytest
from pipecat.frames.frames import TranscriptionFrame, TTSAudioRawFrame, LLMTextFrame
from pipecat.processors.aggregators.llm_context import LLMContext

from app.services.voice_pipeline import (
    BackendWhisperSTTService,
    BackendKokoroTTSService,
    BackendLLMService,
    VoiceSettings,
)


def _fake_wav():
    buf = io.BytesIO()
    sf.write(buf, np.zeros(16, dtype=np.float32), 24000, format="WAV")
    return buf.getvalue()


@pytest.fixture
def patch_services(monkeypatch):
    import app.services.voice_pipeline as vp

    class FakeSpeech:
        def transcribe_audio(self, audio_data, sample_rate=16000):
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
