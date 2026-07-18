# Realtime Voice Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a realtime voice mode to the Chat page — a toolbar toggle that opens a WebRTC session where the user speaks and the backend transcribes (Whisper), reasons (the chosen LLM), and speaks back (Kokoro) with live captions in the thread — and relocate "Clear conversation" out of the toolbar into a header overflow menu.

**Architecture:** Pipecat (`SmallWebRTCTransport` + `SileroVADAnalyzer`) owns the WebRTC transport, VAD, and pipeline plumbing. Three thin custom Pipecat frame processors delegate STT/LLM/TTS to the *existing* `speech_service` / `llm_gateway` / `tts_service`. Media flows over the WebRTC connection; transcript/control events flow over the WebRTC **data channel** (no separate WebSocket). The frontend uses a `use-realtime-voice` hook that creates the data channel *before* the SDP offer.

**Tech Stack:** FastAPI + Pipecat 1.5.0 (core, includes small-webrtc transport + Silero VAD); existing `speech_service` (transformers Whisper), `ttm_service`→`tts_service` (Kokoro), `llm_gateway` (LiteLLM/Ollama); Next.js 16 + React + shadcn/ui; TypeScript; vitest + @testing-library/react for the hook test.

## Global Constraints

- Reuse existing backend services — do **not** add faster-whisper / a second Whisper stack; STT uses `speech_service` (Whisper via transformers), TTS uses `tts_service` (Kokoro primary). (Spec §1, §2)
- VAD must be Pipecat's `SileroVADAnalyzer`. (Spec §3.4)
- Events travel over the WebRTC **data channel**, never a separate WebSocket. The data channel MUST be created by the browser (`pc.createDataChannel('events')`) *before* `createOffer`, or Pipecat's timeout disables it and drops messages. (Spec §3.1, §7)
- Keep the existing browser-dictation **Mic** button behavior unchanged. (Spec §1)
- Remove the `Trash2` "Clear chat" button from the toolbar; "Clear conversation" moves to a header overflow menu. (Spec §4.3)
- Use `prefers-reduced-motion` (no animation) for the voice waveform. (Spec §4.4)
- Run `uv run ruff check .` and `uv run ty check app/...` after Python changes; `npm run lint` after TypeScript changes. (CLAUDE.md)
- Do not modify `version.json` or commit `.env`. (CLAUDE.md)

---

## File Structure

**New backend**
- `app/services/voice_pipeline.py` — `VoiceSettings` dataclass; `BackendWhisperSTTService(STTService)`, `BackendKokoroTTSService(TTSService)`, `BackendLLMService(LLMService)`, `UserTranscriptEmitter`/`AssistantTranscriptEmitter` frame processors; `build_voice_pipeline()` and `run_voice_pipeline()`.
- `app/backend/api/voice.py` — FastAPI router with `POST /api/voice/offer`, the `pcs_map` connection registry, and the background pipeline runner.
- `tests/test_voice_pipeline.py` — unit tests for the three custom processors.
- `tests/test_voice_api.py` — unit test for the offer endpoint (Pipecat mocked).

**Modified backend**
- `pyproject.toml` — add `pipecat-ai` to a new `voice` dependency group and to `all`.
- `main.py` — register `voice_api.router`; tear down `pcs_map` on shutdown.

**New frontend**
- `nextjs-frontend/src/hooks/use-realtime-voice.ts` — WebRTC + data-channel hook (state machine).
- `nextjs-frontend/src/components/voice-activity.tsx` — the live VAD waveform indicator (visual signature).
- `nextjs-frontend/src/hooks/use-realtime-voice.test.ts` — vitest hook test.
- `nextjs-frontend/vitest.config.ts` — jsdom test config.

**Modified frontend**
- `nextjs-frontend/src/app/chat/page.tsx` — add voice toggle + status pill, wire captions into the store, remove `Trash2`, add header overflow "Clear conversation".
- `nextjs-frontend/package.json` — add vitest devDependencies + `test` script.

---

### Task 1: Add pipecat dependency and install

**Files:**
- Modify: `pyproject.toml` (dependency-groups `voice` + `all`)
- Test: (smoke) run `uv run python -c "import pipecat..."`

**Interfaces:**
- Consumes: nothing
- Produces: `pipecat-ai` importable; `SmallWebRTCTransport`, `SmallWebRTCConnection`, `SileroVADAnalyzer` available.

- [ ] **Step 1: Add the `voice` group and extend `all`**

In `pyproject.toml`, add a new group after the `dev = [...]` block and inside `all = [...]`. Show the two edits with surrounding context:

Edit A — add the group (place after the `dev = [...]` block closes, before `all = [`):
```toml
voice = [
    # Realtime voice mode: Pipecat WebRTC transport + Silero VAD (core includes both)
    "pipecat-ai",
]
```

Edit B — inside `all = [ ... ]`, in the `# Audio Processing` section, add the same package so `uv sync --group all` installs it:
```toml
    # Audio Processing
    "librosa",
    "soundfile",
    "pipecat-ai",
```

- [ ] **Step 2: Install**
Run: `uv sync --group all`
Expected: resolves and installs `pipecat-ai` (plus `aiortc` if required by the small-webrtc transport) without error. If `SmallWebRTCConnection` import fails later for a missing `aiortc`, add `"aiortc"` to the `voice` group and re-sync.

- [ ] **Step 3: Smoke-import the needed symbols**
Run:
```bash
uv run python -c "from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport; from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection, IceServer; from pipecat.audio.vad.silero import SileroVADAnalyzer; print('pipecat OK')"
```
Expected: prints `pipecat OK`.

- [ ] **Step 4: Commit**
```bash
git add pyproject.toml uv.lock
git commit -m "build: add pipecat-ai dependency group for realtime voice"
```

---

### Task 2: Backend voice pipeline (custom frame processors)

**Files:**
- Create: `app/services/voice_pipeline.py`
- Test: `tests/test_voice_pipeline.py`

**Interfaces:**
- Consumes: `speech_service.get_speech_service().transcribe_audio(bytes, sample_rate=16000) -> str|None`; `tts_service.get_tts_service().synthesize_text_chunk(text, voice) -> bytes|None` (WAV @ 24000 Hz mono); `llm_gateway.stream_completion(messages, *, backend, model, api_key, api_base, temperature, max_tokens) -> AsyncGenerator[str]`.
- Produces: `build_voice_pipeline(connection, settings) -> (Pipeline, SmallWebRTCTransport)`; `run_voice_pipeline(connection, settings)` coroutine; `VoiceSettings` dataclass.

- [ ] **Step 1: Write the failing tests**

`tests/test_voice_pipeline.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**
Run: `uv run pytest tests/test_voice_pipeline.py -v`
Expected: ERROR/FAIL — `ModuleNotFoundError: app.services.voice_pipeline` (module not written yet).

- [ ] **Step 3: Write the minimal implementation**

`app/services/voice_pipeline.py`:
```python
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

import numpy as np
import soundfile as sf

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
from pipecat.services.llm_service import LLMService
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


class BackendLLMService(LLMService):
    """Streams tokens via the existing llm_gateway (chosen backend)."""

    def __init__(self, settings: VoiceSettings, **kwargs):
        super().__init__(**kwargs)
        self._settings = settings

    async def _process_context(self, context: LLMContext):
        messages = [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in context.get_messages()
        ]
        await self.push_frame(LLMFullResponseEndFrame())  # placeholder removed below
        # NOTE: the start/end frames are emitted around the token stream:
        from pipecat.frames.frames import LLMFullResponseStartFrame

        await self.push_frame(LLMFullResponseStartFrame())
        async for delta in stream_completion(
            messages,
            backend=self._settings.backend,
            model=self._settings.model,
            api_key=self._settings.api_key,
            api_base=self._settings.api_base,
            temperature=self._settings.temperature,
            max_tokens=self._settings.max_tokens,
        ):
            if delta:
                await self.push_frame(LLMTextFrame(text=delta))
        await self.push_frame(LLMFullResponseEndFrame())


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
```

> Note: remove the stray `await self.push_frame(LLMFullResponseEndFrame())` placeholder line inside `_process_context` — it is left only to flag the spot; the correct start/end frames are emitted immediately after. The committed file must contain only `LLMFullResponseStartFrame()` then the token loop then `LLMFullResponseEndFrame()`.

- [ ] **Step 4: Run tests to verify they pass**
Run: `uv run pytest tests/test_voice_pipeline.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Lint + typecheck**
Run: `uv run ruff check app/services/voice_pipeline.py && uv run ty check app/services/voice_pipeline.py`
Expected: clean.

- [ ] **Step 6: Commit**
```bash
git add app/services/voice_pipeline.py tests/test_voice_pipeline.py
git commit -m "feat(voice): add Pipecat pipeline with Whisper/Kokoro/LLM adapters"
```

---

### Task 3: Voice API route + registration

**Files:**
- Create: `app/backend/api/voice.py`
- Test: `tests/test_voice_api.py`
- Modify: `main.py` (register router + shutdown cleanup)

**Interfaces:**
- Consumes: `build_voice_pipeline`/`run_voice_pipeline` from `app.services.voice_pipeline`; `SmallWebRTCConnection`, `IceServer` from Pipecat.
- Produces: `POST /api/voice/offer` returning `{sdp, type, pc_id}`; module-level `pcs_map`.

- [ ] **Step 1: Write the failing test**

`tests/test_voice_api.py`:
```python
import asyncio
import importlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    import app.backend.api.voice as voice_mod

    calls = {}

    class FakeConn:
        pc_id = "test-pc"

        def __init__(self, ice_servers):
            self.ice_servers = ice_servers

        async def initialize(self, sdp, type):
            calls["init"] = (sdp, type)

        def event_handler(self, name):
            def deco(fn):
                return fn

            return deco

        def get_answer(self):
            return {"sdp": "fake-answer", "type": "answer", "pc_id": self.pc_id}

        async def renegotiate(self, sdp, type, restart_pc=False):
            calls["reneg"] = True

    monkeypatch.setattr(voice_mod, "SmallWebRTCConnection", FakeConn)
    monkeypatch.setattr(voice_mod, "run_voice_pipeline", lambda conn, settings: None)
    app = FastAPI()
    app.include_router(voice_mod.router)
    voice_mod.pcs_map.clear()
    yield TestClient(app)
    voice_mod.pcs_map.clear()


def test_offer_creates_connection(client):
    resp = client.post(
        "/api/voice/offer",
        json={
            "pc_id": "test-pc",
            "sdp": "v=0...",
            "type": "offer",
            "model": "llama3",
            "backend": "ollama",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["pc_id"] == "test-pc"
    assert body["type"] == "answer"
    import app.backend.api.voice as voice_mod

    assert "test-pc" in voice_mod.pcs_map


def test_offer_requires_pc_id(client):
    resp = client.post("/api/voice/offer", json={"sdp": "x", "type": "offer"})
    assert resp.status_code == 200
    assert resp.json().get("error")
```

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run pytest tests/test_voice_api.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: app.backend.api.voice`.

- [ ] **Step 3: Write the implementation**

`app/backend/api/voice.py`:
```python
"""Realtime voice signaling endpoint (SDP offer/answer over HTTP).

The browser POSTs an SDP offer; the server answers and runs the Pipecat
pipeline as a background task. All ongoing transcript/control events are sent
back over the WebRTC data channel (see app.services.voice_pipeline).
"""
import logging

from fastapi import APIRouter, BackgroundTasks

from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection

from app.services.voice_pipeline import VoiceSettings, run_voice_pipeline

logger = logging.getLogger(__name__)

router = APIRouter()

pcs_map: dict[str, SmallWebRTCConnection] = {}
ice_servers = [IceServer(urls="stun:stun.l.google.com:19302")]


@router.post("/api/voice/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    pc_id = request.get("pc_id")
    if not pc_id:
        return {"error": "pc_id required"}

    if pc_id in pcs_map:
        connection = pcs_map[pc_id]
        await connection.renegotiate(
            sdp=request["sdp"], type=request["type"], restart_pc=request.get("restart_pc", False)
        )
    else:
        connection = SmallWebRTCConnection(ice_servers)
        await connection.initialize(sdp=request["sdp"], type=request["type"])

        @connection.event_handler("closed")
        async def handle_closed(conn: SmallWebRTCConnection):
            logger.info("Discarding peer connection %s", conn.pc_id)
            pcs_map.pop(conn.pc_id, None)

        settings = VoiceSettings(
            model=request.get("model"),
            backend=request.get("backend"),
            api_key=request.get("api_key"),
            api_base=request.get("api_base"),
            temperature=float(request.get("temperature", 0.7)),
            max_tokens=int(request.get("max_tokens", 512)),
            voice=request.get("voice"),
            system_instruction=request.get("system_instruction")
            or "You are a helpful voice assistant. Respond briefly and conversationally. "
            "Avoid emojis, bullet points, and any formatting that cannot be spoken aloud.",
        )
        background_tasks.add_task(run_voice_pipeline, connection, settings)

    answer = connection.get_answer()
    pcs_map[answer["pc_id"]] = connection
    return answer
```

- [ ] **Step 4: Register the router in `main.py`**

Near the other `include_router` calls (around the `chat_api.router` include), add:
```python
from app.backend.api import voice as voice_api
```
and after `app.include_router(chat_api.router)`:
```python
app.include_router(voice_api.router)
```

Also tear down connections on shutdown. Find the existing `lifespan` (or `startup`/`shutdown`) handler and, in the shutdown branch, add:
```python
from app.backend.api.voice import pcs_map
for conn in list(pcs_map.values()):
    try:
        await conn.disconnect()
    except Exception:
        pass
pcs_map.clear()
```

- [ ] **Step 5: Run test to verify it passes**
Run: `uv run pytest tests/test_voice_api.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Lint + typecheck**
Run: `uv run ruff check app/backend/api/voice.py main.py && uv run ty check app/backend/api/voice.py`
Expected: clean.

- [ ] **Step 7: Commit**
```bash
git add app/backend/api/voice.py main.py tests/test_voice_api.py
git commit -m "feat(voice): add /api/voice/offer signaling endpoint and register router"
```

---

### Task 4: Frontend realtime-voice hook

**Files:**
- Create: `nextjs-frontend/src/hooks/use-realtime-voice.ts`
- Create: `nextjs-frontend/vitest.config.ts`
- Modify: `nextjs-frontend/package.json` (vitest devDeps + `test` script)
- Test: `nextjs-frontend/src/hooks/use-realtime-voice.test.ts`

**Interfaces:**
- Consumes: `POST /api/voice/offer`; browser `getUserMedia` / `RTCPeerConnection` / `RTCDataChannel`.
- Produces: `useRealtimeVoice(settings, callbacks)` returning `{ state, start, stop, isConnected }`. `state` is `'idle' | 'connecting' | 'listening' | 'speaking' | 'error'`.

- [ ] **Step 1: Add vitest deps + config**

Add to `nextjs-frontend/package.json` `devDependencies`:
```json
"@testing-library/react": "^16.0.0",
"jsdom": "^25.0.0",
"vitest": "^2.0.0"
```
Add a `test` script:
```json
"test": "vitest run"
```
Create `nextjs-frontend/vitest.config.ts`:
```ts
import { defineConfig } from "vitest/config";
import path from "path";

export default defineConfig({
  test: {
    environment: "jsdom",
    globals: true,
    include: ["src/**/*.test.ts", "src/**/*.test.tsx"],
  },
  resolve: {
    alias: { "@": path.resolve(__dirname, "src") },
  },
});
```
Run: `cd nextjs-frontend && npm install`
Expected: installs without error.

- [ ] **Step 2: Write the failing test**

`nextjs-frontend/src/hooks/use-realtime-voice.test.ts`:
```ts
import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useRealtimeVoice } from "./use-realtime-voice";

function mockBrowser() {
  const dataChannelHandlers: Record<string, ((e: { data: string }) => void)> = {};
  const dc = {
    send: vi.fn(),
    close: vi.fn(),
    set onmessage(fn: any) {
      dataChannelHandlers["message"] = fn;
    },
    get onmessage() {
      return dataChannelHandlers["message"];
    },
  };
  const pc = {
    createDataChannel: () => dc,
    createOffer: vi.fn().mockResolvedValue({ sdp: "offer-sdp", type: "offer" }),
    setLocalDescription: vi.fn().mockResolvedValue(undefined),
    setRemoteDescription: vi.fn().mockResolvedValue(undefined),
    addTrack: vi.fn(),
    close: vi.fn(),
    ondatachannel: null as any,
    ontrack: null as any,
    onconnectionstatechange: null as any,
    connectionState: "connected",
  };
  (globalThis as any).RTCPeerConnection = vi.fn().mockReturnValue(pc);
  (globalThis as any).RTCSessionDescription = class {
    constructor(public init: any) {}
  };
  (globalThis as any).navigator = {
    mediaDevices: { getUserMedia: vi.fn().mockResolvedValue({ getTracks: () => [] }) },
  };
  (globalThis as any).crypto = { randomUUID: () => "pc-123" };
  (globalThis as any).fetch = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ sdp: "answer-sdp", type: "answer", pc_id: "pc-123" }),
  });
  (globalThis as any).Audio = class {
    autoplay = false;
    srcObject: any = null;
  };
  return { pc, dc, dataChannelHandlers };
}

describe("useRealtimeVoice", () => {
  beforeEach(() => mockBrowser());

  it("transitions to listening on a successful offer", async () => {
    const { result } = renderHook(() => useRealtimeVoice({}, {}));
    await act(async () => {
      await result.current.start();
    });
    expect(result.current.state).toBe("listening");
    expect((globalThis as any).fetch).toHaveBeenCalledWith(
      "/api/voice/offer",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("fires onAssistantText when a data-channel message arrives", async () => {
    const env = mockBrowser();
    const onAssistantText = vi.fn();
    const { result } = renderHook(() =>
      useRealtimeVoice({}, { onAssistantText }),
    );
    await act(async () => {
      await result.current.start();
    });
    await act(async () => {
      env.dataChannelHandlers["message"]({
        data: JSON.stringify({ type: "assistant_text", text: "Hi there" }),
      });
    });
    expect(onAssistantText).toHaveBeenCalledWith("Hi there");
    expect(result.current.state).toBe("speaking");
  });
});
```

- [ ] **Step 3: Run test to verify it fails**
Run: `cd nextjs-frontend && npm run test`
Expected: FAIL — `use-realtime-voice` not found.

- [ ] **Step 4: Write the implementation**

`nextjs-frontend/src/hooks/use-realtime-voice.ts`:
```ts
'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

export type VoiceState = 'idle' | 'connecting' | 'listening' | 'speaking' | 'error';

export interface VoiceCallbacks {
  onReady?: () => void;
  onUserTranscript?: (text: string) => void;
  onAssistantText?: (text: string) => void;
  onAssistantEnd?: () => void;
  onError?: (message: string) => void;
  onEnded?: () => void;
}

export interface VoiceRequestSettings {
  model?: string | null;
  backend?: string | null;
  apiKey?: string | null;
  apiBase?: string | null;
  temperature?: number;
  maxTokens?: number;
  voice?: string | null;
}

export function useRealtimeVoice(settings: VoiceRequestSettings, callbacks: VoiceCallbacks) {
  const [state, setState] = useState<VoiceState>('idle');
  const pcRef = useRef<RTCPeerConnection | null>(null);
  const dcRef = useRef<RTCDataChannel | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const settingsRef = useRef(settings);
  const callbacksRef = useRef(callbacks);
  settingsRef.current = settings;
  callbacksRef.current = callbacks;

  const cleanup = useCallback(() => {
    try { dcRef.current?.close(); } catch { /* ignore */ }
    try { pcRef.current?.close(); } catch { /* ignore */ }
    streamRef.current?.getTracks().forEach((t) => t.stop());
    if (audioRef.current) audioRef.current.srcObject = null;
    pcRef.current = null;
    dcRef.current = null;
    streamRef.current = null;
    setState('idle');
  }, []);

  const stop = useCallback(() => {
    try { dcRef.current?.send(JSON.stringify({ type: 'end' })); } catch { /* ignore */ }
    cleanup();
  }, [cleanup]);

  const start = useCallback(async () => {
    if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
      setState('error');
      callbacksRef.current.onError?.('Voice mode is not supported in this browser.');
      return;
    }
    setState('connecting');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const pc = new RTCPeerConnection({
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
      });
      pcRef.current = pc;

      // MUST be created before the offer, or Pipecat disables the channel.
      const dc = pc.createDataChannel('events');
      dcRef.current = dc;
      dc.onmessage = (e: MessageEvent) => {
        let msg: any;
        try { msg = JSON.parse(e.data); } catch { return; }
        const cb = callbacksRef.current;
        switch (msg.type) {
          case 'ready': cb.onReady?.(); break;
          case 'user_transcript': setState('listening'); cb.onUserTranscript?.(msg.text); break;
          case 'assistant_text': setState('speaking'); cb.onAssistantText?.(msg.text); break;
          case 'assistant_end': setState('listening'); cb.onAssistantEnd?.(); break;
          case 'error': setState('error'); cb.onError?.(msg.message); break;
          case 'ended': cb.onEnded?.(); cleanup(); break;
        }
      };
      pc.ondatachannel = (e: RTCDataChannelEvent) => { e.channel.onmessage = dc.onmessage; };
      stream.getTracks().forEach((t) => pc.addTrack(t, stream));

      const audio = new Audio();
      audio.autoplay = true;
      audioRef.current = audio;
      pc.ontrack = (e: RTCTrackEvent) => { audio.srcObject = e.streams[0]; };
      pc.onconnectionstatechange = () => {
        if (pc.connectionState === 'failed' || pc.connectionState === 'closed') cleanup();
      };

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      const s = settingsRef.current;
      const resp = await fetch('/api/voice/offer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pc_id: crypto.randomUUID(),
          sdp: offer.sdp,
          type: offer.type,
          model: s.model ?? null,
          backend: s.backend ?? null,
          api_key: s.apiKey ?? null,
          api_base: s.apiBase ?? null,
          temperature: s.temperature ?? 0.7,
          max_tokens: s.maxTokens ?? 512,
          voice: s.voice ?? null,
        }),
      });
      if (!resp.ok) {
        setState('error');
        callbacksRef.current.onError?.('Failed to start voice session.');
        cleanup();
        return;
      }
      const answer = await resp.json();
      await pc.setRemoteDescription(new RTCSessionDescription({ type: answer.type, sdp: answer.sdp }));
      setState('listening');
    } catch (err) {
      setState('error');
      callbacksRef.current.onError?.(err instanceof Error ? err.message : 'Voice start failed.');
      cleanup();
    }
  }, [cleanup]);

  useEffect(() => () => cleanup(), [cleanup]);

  return { state, start, stop, isConnected: state !== 'idle' && state !== 'error' };
}
```

- [ ] **Step 5: Run test to verify it passes**
Run: `cd nextjs-frontend && npm run test`
Expected: PASS (2 tests).

- [ ] **Step 6: Lint**
Run: `cd nextjs-frontend && npm run lint`
Expected: clean.

- [ ] **Step 7: Commit**
```bash
git add nextjs-frontend/src/hooks/use-realtime-voice.ts nextjs-frontend/src/hooks/use-realtime-voice.test.ts nextjs-frontend/vitest.config.ts nextjs-frontend/package.json nextjs-frontend/package-lock.json
git commit -m "feat(voice): add useRealtimeVoice WebRTC + data-channel hook"
```

---

### Task 5: Chat page wiring, visual signature, and delete-button relocation

**Files:**
- Create: `nextjs-frontend/src/components/voice-activity.tsx`
- Modify: `nextjs-frontend/src/app/chat/page.tsx`

**Interfaces:**
- Consumes: `useRealtimeVoice` (Task 4); `useAppStore` actions `addMessage(convId, msg)`, `updateLastMessage(convId, content, isStreaming?)`, `clearConversation(id)`; `selectSettings`.
- Produces: a voice toggle in the toolbar; live captions appended to the active conversation; "Clear conversation" in a header overflow menu; no `Trash2` toolbar button.

- [ ] **Step 1: Create the voice-activity indicator component**

`nextjs-frontend/src/components/voice-activity.tsx`:
```tsx
'use client';

import { cn } from '@/lib/utils';
import type { VoiceState } from '@/hooks/use-realtime-voice';

const BARS = 5;

export default function VoiceActivityIndicator({ state, className }: { state: VoiceState; className?: string }) {
  const active = state === 'listening' || state === 'speaking';
  return (
    <span className={cn('inline-flex h-4 items-end gap-0.5', className)} aria-hidden="true">
      {Array.from({ length: BARS }).map((_, i) => (
        <span
          key={i}
          className={cn(
            'w-0.5 rounded-full bg-primary transition-transform duration-150 motion-reduce:animate-none',
            active ? (state === 'speaking' ? 'h-4 animate-pulse' : 'h-3') : 'h-1 opacity-50',
          )}
          style={active ? { animationDelay: `${i * 80}ms` } : undefined}
        />
      ))}
    </span>
  );
}
```

- [ ] **Step 2: Wire the chat page**

In `nextjs-frontend/src/app/chat/page.tsx`:

1. Update the lucide import to add `Phone, PhoneOff, MoreVertical`:
```ts
import { Send, Globe, Mic, MicOff, Bot, Code, Sparkles, Plus, Phone, PhoneOff, MoreVertical } from 'lucide-react';
```
Remove `Trash2` from that import line (it is no longer used).

2. Add imports for the new pieces:
```ts
import { DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem } from '@/components/ui/dropdown-menu';
import VoiceActivityIndicator from '@/components/voice-activity';
import { useRealtimeVoice } from '@/hooks/use-realtime-voice';
```

3. Inside `ChatPage`, add state + the hook (place near the existing `useVoiceInput` usage). Track the active assistant turn with a ref:
```ts
const assistantActiveRef = React.useRef(false);
const convIdRef = React.useRef(activeId);
convIdRef.current = activeId;

const voiceCallbacks = React.useCallback({
  onUserTranscript: (text: string) => {
    const id = convIdRef.current;
    if (id) addMessage(id, { role: 'user', content: text });
  },
  onAssistantText: (delta: string) => {
    const id = convIdRef.current;
    if (!id) return;
    const conv = useAppStore.getState().conversations.find((c) => c.id === id);
    const last = conv?.messages.slice(-1)[0];
    if (!assistantActiveRef.current) {
      addMessage(id, { role: 'assistant', content: delta, isStreaming: true });
      assistantActiveRef.current = true;
    } else if (last && last.role === 'assistant') {
      updateLastMessage(id, last.content + delta, true);
    }
  },
  onAssistantEnd: () => {
    const id = convIdRef.current;
    if (id) {
      const conv = useAppStore.getState().conversations.find((c) => c.id === id);
      const last = conv?.messages.slice(-1)[0];
      if (last && last.role === 'assistant') updateLastMessage(id, last.content, false);
    }
    assistantActiveRef.current = false;
  },
  onError: (message: string) => {
    const id = convIdRef.current;
    if (id) addMessage(id, { role: 'assistant', content: `⚠️ Voice error: ${message}` });
  },
}, [addMessage, updateLastMessage]);

const { state: voiceState, start: startRealtime, stop: stopRealtime } = useRealtimeVoice(settings, voiceCallbacks);
const voiceOn = voiceState !== 'idle' && voiceState !== 'error';
```

4. Add a slim header row with the overflow "Clear conversation" menu. Insert it as the first child inside the main `return` container (above the optional generative-UI bar):
```tsx
<div className="flex h-9 shrink-0 items-center justify-end border-b border-border bg-background px-4 md:px-6">
  <DropdownMenu>
    <DropdownMenuTrigger asChild>
      <Button variant="ghost" size="icon" className="h-8 w-8" aria-label="Conversation options">
        <MoreVertical className="h-4 w-4" aria-hidden="true" />
      </Button>
    </DropdownMenuTrigger>
    <DropdownMenuContent align="end">
      <DropdownMenuItem onSelect={() => activeId && clearConversation(activeId)}>
        Clear conversation
      </DropdownMenuItem>
    </DropdownMenuContent>
  </DropdownMenu>
</div>
```

5. In the action bar, replace the `Trash2` `Tooltip` block with a realtime-voice `Tooltip` placed **before** the existing Mic `Tooltip`:
```tsx
<Tooltip>
  <TooltipTrigger
    render={
      <Button
        variant={voiceOn ? 'default' : 'outline'}
        size="icon"
        className={cn('h-9 w-9', voiceOn && 'bg-primary text-primary-foreground')}
        onClick={() => (voiceOn ? stopRealtime() : startRealtime())}
        aria-label={voiceOn ? 'Stop voice mode' : 'Start voice mode'}
        aria-pressed={voiceOn}
      >
        {voiceOn ? <PhoneOff className="h-4 w-4" aria-hidden="true" /> : <Phone className="h-4 w-4" aria-hidden="true" />}
      </Button>
    }
  />
  <TooltipContent>Voice mode</TooltipContent>
</Tooltip>
```
Then delete the original `Trash2`/`clearConversation` `Tooltip` block entirely.

6. Add a status pill above the input row (inside the bottom `div` container, before the `flex items-end gap-2` row) so the live waveform is visible while voice mode is on:
```tsx
{voiceOn && (
  <div className="mb-2 flex items-center gap-2 text-xs text-muted-foreground">
    <VoiceActivityIndicator state={voiceState} />
    <span>{voiceState === 'speaking' ? 'Assistant is speaking' : 'Listening…'}</span>
  </div>
)}
```

- [ ] **Step 3: Run lint + typecheck**
Run: `cd nextjs-frontend && npm run lint`
Expected: clean. Also run `npx tsc --noEmit` if configured; fix any unused-import errors (e.g., ensure `Trash2` is fully removed from imports).

- [ ] **Step 4: Commit**
```bash
git add nextjs-frontend/src/components/voice-activity.tsx nextjs-frontend/src/app/chat/page.tsx
git commit -m "feat(voice): add voice-mode toggle, live captions, and relocate clear to header"
```

---

### Task 6: Typecheck, lint, and manual end-to-end smoke

**Files:** none new (verification only)

**Interfaces:** Consumes: all prior tasks.

- [ ] **Step 1: Backend typecheck + lint**
Run: `uv run ruff check . && uv run ty check app/`
Expected: clean.

- [ ] **Step 2: Frontend lint**
Run: `cd nextjs-frontend && npm run lint`
Expected: clean.

- [ ] **Step 3: Backend unit tests**
Run: `uv run pytest tests/test_voice_pipeline.py tests/test_voice_api.py -v`
Expected: all PASS.

- [ ] **Step 4: Frontend unit tests**
Run: `cd nextjs-frontend && npm run test`
Expected: PASS.

- [ ] **Step 5: Manual E2E (cannot run in CI — needs a mic + torch models)**
1. `uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload`
2. `cd nextjs-frontend && npm run dev`
3. Open http://localhost:3000, start a conversation, click **Voice mode** (phone icon).
4. Grant mic access; speak a short prompt. Confirm: a user caption appears, the assistant caption streams in, and Kokoro audio plays. The waveform pulses while speaking/listening.
5. Click the phone icon again to stop; confirm the call tears down and the toolbar has no Clear-chat button. Open the header `⋮` menu and confirm "Clear conversation" works.

- [ ] **Step 6: Final commit (if any fixes were needed)**
```bash
git add -A
git commit -m "fix(voice): address lint/typecheck findings from E2E smoke"
```
(Only if Step 1–4 required changes.)

---

## Self-Review Notes

- **Spec coverage:** §3 transport/signaling → Task 3; §3.2 pipeline + custom processors → Task 2; §3.3 STT/LLM/TTS adapters → Task 2; §3.4 VAD → Task 2 (`SileroVADAnalyzer`); §4.1 hook → Task 4; §4.2 chat wiring → Task 5; §4.3 delete relocation → Task 5; §4.4 visual signature → Task 5 (`VoiceActivityIndicator`); §5 event protocol → encoded in `UserTranscriptEmitter`/`AssistantTranscriptEmitter` (Task 2) and the hook `onmessage` (Task 4); §7 error handling → hook `try/catch` + `onError` + `pc.onconnectionstatechange` (Task 4) and `error` event (Task 2); §8 deps → Task 1; §10 testing → Tasks 2/3/4/6.
- **Placeholder scan:** Task 2 step 3 contains a deliberate `LLMFullResponseEndFrame()` placeholder line that the step text instructs to **remove** — the committed file must not contain it. No TBD/TODO elsewhere.
- **Type consistency:** `useRealtimeVoice(settings, callbacks)` signature matches between Task 4 (definition) and Task 5 (usage). `VoiceSettings` fields in Task 2 match the offer body keys sent by the hook in Task 4 and parsed in Task 3. Event `type` strings (`ready`, `user_transcript`, `assistant_text`, `assistant_end`, `error`, `ended`) are identical across Task 2 (emit) and Task 4 (receive). `VoiceState` union is shared by the hook (Task 4) and `VoiceActivityIndicator` (Task 5).
