"""Tests for SpeechService.transcribe_pcm (realtime raw-PCM path).

The realtime voice pipeline feeds Pipecat ``AudioRawFrame`` PCM bytes (16-bit
mono) into STT. ``transcribe_pcm`` must decode that PCM directly rather than
handing the bytes to ``librosa.load`` as if they were an encoded file (which
previously produced ``PySoundFile failed`` and garbage transcriptions).
"""

import numpy as np
import pytest

from app.services.speech_service import SpeechService


class _FakeWhisperPipe:
    """Stands in for the transformers ASR pipeline.

    Asserts the contract Whisper expects: a float32 array in [-1, 1] at 16 kHz,
    and returns a canned transcription so no model is loaded.
    """

    def __init__(self):
        self.last_audio = None
        self.last_sampling_rate = None

    def __call__(self, payload: dict):
        self.last_audio = payload["raw"]
        self.last_sampling_rate = payload["sampling_rate"]
        return {"text": "decoded ok"}


@pytest.fixture
def stubbed_service(monkeypatch):
    svc = SpeechService()
    svc._model_loaded = True
    svc.pipe = _FakeWhisperPipe()
    return svc


def _int16_pcm(signal: np.ndarray, rate: int = 16000) -> bytes:
    """Render a float signal ([-1, 1]) as 16-bit mono PCM bytes at ``rate``."""
    pcm = np.clip(signal, -1.0, 1.0)
    return (pcm * 32767.0).astype(np.int16).tobytes()


async def test_transcribe_pcm_decodes_raw_int16(stubbed_service):
    # A simple tone; any signal proves we decode PCM instead of loading a file.
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    signal = np.sin(2 * np.pi * 220 * t) * 0.3
    pcm = _int16_pcm(signal, rate=16000)

    text = stubbed_service.transcribe_pcm(pcm, source_rate=16000)

    assert text == "decoded ok"
    # Pipe received a float32 array normalized into [-1, 1].
    assert stubbed_service.pipe.last_audio.dtype == np.float32
    assert stubbed_service.pipe.last_sampling_rate == 16000
    assert float(stubbed_service.pipe.last_audio.min()) >= -1.0
    assert float(stubbed_service.pipe.last_audio.max()) <= 1.0


async def test_transcribe_pcm_resamples_when_source_rate_differs(stubbed_service, monkeypatch):
    # If upstream audio isn't 16 kHz, transcribe_pcm must resample to 16 kHz.
    import app.services.speech_service as ss

    # Stub librosa.resample to avoid pulling in the heavy dependency path.
    monkeypatch.setattr(ss, "librosa", type("L", (), {"resample": staticmethod(lambda a, orig_sr, target_sr: a)})())

    t = np.linspace(0, 1.0, 48000, endpoint=False)
    signal = np.sin(2 * np.pi * 220 * t) * 0.3
    pcm = _int16_pcm(signal, rate=48000)

    text = stubbed_service.transcribe_pcm(pcm, source_rate=48000)

    assert text == "decoded ok"
    assert stubbed_service.pipe.last_sampling_rate == 16000


async def test_transcribe_pcm_empty_audio_returns_none(stubbed_service):
    assert stubbed_service.transcribe_pcm(b"", source_rate=16000) is None
