# Realtime Voice Mode — Design Spec

- **Date:** 2026-07-18
- **Status:** Approved (design); pending implementation plan
- **Scope:** `nextjs-frontend/` (React/Next.js 16) + FastAPI backend (`app/`)
- **Owner:** debabratamishra

## 1. Goal & scope

Add a **realtime voice mode** to the Chat page:

- A **toggle in the chat action bar** opens a WebRTC session. The user speaks; the
  backend transcribes (Whisper), reasons (the user's chosen LLM), and speaks back
  (Kokoro), with **live captions streaming into the same chat thread**.
- **Remove** the Clear-chat (`Trash2`) button from the action bar and **relocate**
  "Clear conversation" to a header overflow menu.
- Keep the existing browser-dictation **Mic** button exactly as-is.
- Backend uses **WebRTC (Pipecat `SmallWebRTCTransport`) + Silero VAD (Pipecat)**,
  and a **sandwich** `STT → LLM → TTS` that delegates to the *existing*
  `speech_service` / `llm_gateway` / `tts_service`.

### Explicitly in scope
- Voice-mode toggle button + status/visual indicator in the chat bar.
- WebRTC media (mic in, speaker out) + WebRTC **data channel** for events/control.
- Three thin custom Pipecat frame processors wrapping existing backend services.
- Live captioning of both sides into the conversation store / chat thread.
- Relocation of "Clear conversation" to a header overflow menu.

### Out of scope (see §9)
- Daily/cloud transport, multi-user rooms, SIP/telephony.
- Replacing the existing browser-dictation Mic behavior.
- Server-side audio logging / conversation persistence of raw audio.

## 2. Approach decision

**Chosen — Hybrid Pipecat pipeline.** Pipecat owns transport + VAD + the
pipeline/frame plumbing. Three thin **custom frame processors** wrap our existing
backend services so STT/LLM/TTS are still done "through the backend" by the code
that already exists, with one source of truth for model + keys (the chat
`settings`).

**Rejected — pure native Pipecat adapters** (`FasterWhisperSTTService`,
`KokoroTTSService`, `LiteLLMLLMService`): would re-add dependencies that were
intentionally dropped (`CLAUDE.md` notes faster-whisper was removed in favor of
transformers Whisper) and would bypass `llm_gateway`. Lower glue risk, but
duplicates logic and ignores the established services layer.

**Fallback (LLM leg only):** if the custom `BackendLLMService` wrapper around
`llm_gateway` proves fragile against Pipecat's `LLMService` base-class contract,
fall back to Pipecat's native `LiteLLMLLMService` (extra `pipecat-ai-litellm`)
driven by the same `settings.model/backend/api_key/api_base`. `llm_gateway` is
itself a thin LiteLLM/ollama wrapper, so behavior is equivalent.

## 3. Backend design

### 3.1 Transport & signaling
- Integrate into the **existing FastAPI app** (`main.py`) on port 8000 — no second
  server process.
- **`POST /api/voice/offer`** — mirrors the canonical Pipecat small-webrtc pattern:
  body `{ pc_id, sdp, type, restart_pc?, model, backend, api_key, api_base,
  temperature, max_tokens, system_instruction?, voice? }`. Returns the SDP answer
  `{ sdp, type, pc_id }`. On first offer it creates a `SmallWebRTCConnection`
  (ICE = Google STUN `stun:stun.l.google.com:19302`), initializes it, registers a
  `closed` handler to evict from the pc map, and schedules the pipeline as a
  background task. On a re-offer for an existing `pc_id` it calls `renegotiate`.
- **No separate WebSocket.** The ongoing transcript/event stream travels over the
  WebRTC **data channel** created by the browser before the offer. The server pushes
  JSON via `SmallWebRTCConnection.send_app_message(msg)`; the browser listens on the
  channel's `onmessage`. This removes the need for any cross-request event bus.

### 3.2 New module `app/services/voice_pipeline.py`
`build_voice_pipeline(connection: SmallWebRTCConnection, settings: VoiceSettings)`
constructs:

```
Pipeline([
    transport.input(),                       # mic audio in
    BackendWhisperSTTService(...),           # -> speech_service
    user_aggregator(vad=SileroVADAnalyzer()),# VAD-gated user turn
    BackendLLMService(...),                  # -> llm_gateway
    BackendKokoroTTSService(...),            # -> tts_service
    transport.output(),                      # speaker audio out
    assistant_aggregator,
    TranscriptEmitter(connection),           # pushes events on data channel
])
```

Plus a `PipelineWorker` and a `WorkerRunner` (same pattern as the Pipecat example).
`run_example` wires `on_client_connected` (kick off with an optional greeting via
`LLMRunFrame`) and `on_client_disconnected` (cancel worker).

### 3.3 Custom frame processors (delegate to existing services)
- **`BackendWhisperSTTService(STTService)`** — override `run_stt(self, audio: bytes)`
  → `speech_service.transcribe_audio(audio, sample_rate=16000)` → yield a
  `TranscriptionFrame(text, user_id="user", timestamp=..., finalized=True)`.
- **`BackendKokoroTTSService(TTSService)`** — override
  `run_tts(self, text: str, context_id: str)` →
  `tts_service.synthesize_text_chunk(text, voice)` → yield `TTSAudioRawFrame(
  audio, sample_rate, num_channels)`.
- **`BackendLLMService(LLMService)`** — stream tokens via `llm_gateway` using the
  settings from the offer body; yield `TextFrame` deltas (and the standard
  `LLMFullResponseStart/EndFrame`s) so the aggregator + TTS + emitter work normally.
- **`TranscriptEmitter(FrameProcessor)`** — injected with the bound
  `connection.send_app_message`. Emits on the data channel:
  - `TranscriptionFrame` → `user_transcript` event,
  - `TextFrame` (assistant) → `assistant_text` delta event,
  - `UserStoppedSpeakingFrame` / `BotStartedSpeakingFrame` / `BotStoppedSpeakingFrame`
    → `turn` state event.

All three custom services must be verified against the installed Pipecat 1.5.0
base-class signatures (`STTService.run_stt`, `TTSService.run_tts`,
`LLMService.run_llm` / context handling) before/at implementation time. Frame
constructors confirmed: `TranscriptionFrame(text, user_id, timestamp, language=None,
result=None, finalized=False)`, `TTSAudioRawFrame(audio, sample_rate, num_channels,
context_id=None)`, `TextFrame(text, ...)`.

### 3.4 VAD
`SileroVADAnalyzer()` is passed into the `LLMContextAggregatorPair` user params.
This is the "VAD thru pipecat" requirement and drives turn detection + barge-in
(interrupting TTS when the user starts speaking — handled by the aggregator pair).
The VAD model downloads on first use (cold-start latency noted in §9).

### 3.5 Session/connection registry
Module-level `pcs_map: dict[str, SmallWebRTCConnection]` keyed by `pc_id`, exactly
like the Pipecat example, evicted on `closed`. No separate event bus.

## 4. Frontend design

### 4.1 New hook `nextjs-frontend/src/hooks/use-realtime-voice.ts` (`'use client'`)
State machine: `idle → connecting → listening → speaking → error`.

`start(settings)`:
1. `getUserMedia({ audio: true })`; bail to `error` if denied/unsupported.
2. `new RTCPeerConnection({ iceServers:[{urls:'stun:stun.l.google.com:19302'}] })`.
3. **`const dc = pc.createDataChannel('events')` BEFORE `createOffer`** (required —
   Pipecat's `SmallWebRTCConnection` receives the channel via `on("datachannel")`;
   if absent, its timeout disables the channel and drops messages).
4. `dc.onmessage` → parse JSON → route to callbacks (`onUserTranscript`,
   `onAssistantText`, `onTurnChange`, `onError`, `onReady`, `onEnded`).
5. Add local mic track; `createOffer` → `setLocalDescription` →
   `POST /api/voice/offer` with `{pc_id, sdp, type, ...settings}` →
   `setRemoteDescription(answer)`.
6. Attach remote track to a hidden `<audio autoplay>` element for playback.

`stop()` tears down: close `dc`, `pc.close()`, stop mic tracks, and send a `dc`
control message `{type:'end'}` (best-effort) before closing.

Exposes: `state`, `isConnected`, `start`, `stop`. Silently unsupported (no button /
disabled) when `getUserMedia`/RTCPeerConnection are absent.

### 4.2 Chat page wiring (`nextjs-frontend/src/app/chat/page.tsx`)
- **Add a Voice-mode toggle** button next to the existing Mic (icon `Phone` /
  `PhoneOff`). Toggling on calls `start(settings)`; off calls `stop()`.
- Callbacks feed the **existing conversation store**:
  - `onUserTranscript(text)` → append a user message.
  - `onAssistantText(delta)` → create the assistant message on first delta, then
    append (live caption) as it streams; finalize when `turn` returns to `listening`.
  - `onTurnChange(state)` → update a small status pill (`Listening` / `Speaking` /
    `Thinking`).
- The **existing Mic dictation**, text input, web-search toggle, and `isStreaming`
  text send remain fully usable and independent of voice mode.

### 4.3 Delete-button removal + relocation
- Remove the `Trash2` "Clear chat" `Button` from the action bar.
- Add a **header overflow menu** (`MoreVertical`) in the chat view (a slim header
  row above the thread, or reuse the existing top region) containing
  **"Clear conversation"** → `clearConversation(activeId)` from the store.

### 4.4 Visual signature (frontend-design)
Because VAD is the heart of this feature, the voice-mode **on** state shows a
**live voice-activity waveform** — a compact row of bars that animates with the
`turn`/`listening`/`speaking` events — living in the status pill. Everything else
stays quiet; it uses the app's existing `primary` token to remain consistent with
the shadcn/ui design system. Respects `prefers-reduced-motion` (static bars, no
pulse). This is the one memorable element; no other decoration is added.

## 5. Event protocol (data channel, JSON)

Backend → browser:
| `type` | fields | meaning |
|--------|--------|---------|
| `ready` | — | pipeline started; safe to speak |
| `user_transcript` | `text`, `final` | finalized user speech |
| `assistant_text` | `text` (delta), `final` | streamed assistant caption |
| `turn` | `state`: `listening`\|`speaking`\|`thinking`\|`idle` | VAD/LLM turn state |
| `error` | `message` | backend error (e.g., missing key/model) |
| `ended` | — | session closed by backend |

Browser → backend (control, over same data channel):
| `type` | fields | meaning |
|--------|--------|---------|
| `end` | — | hang up |
| `mute` | `value: bool` | toggle mic mute |

## 6. Data / turn flow
1. User speaks → Silero VAD detects speech start/stop → `BackendWhisperSTTService`
   transcribes → `TranscriptEmitter` sends `user_transcript` → UI appends user bubble;
   `turn` → `listening` until speech ends.
2. User turn finalizes → `BackendLLMService` streams tokens →
   `assistant_aggregator` → `TranscriptEmitter` sends `assistant_text` deltas → UI
   appends/updates assistant bubble (caption) **and** `BackendKokoroTTSService`
   synthesizes → `transport.output()` plays audio; `turn` → `speaking`.
3. `turn` returns to `listening` when TTS completes / next speech starts.

## 7. Error handling & edge cases
- Mic permission denied / no WebRTC → button disabled or inline error; chat never
  crashes.
- Backend model/key missing → pipeline logs + emits `error`; UI shows a
  non-blocking toast and reverts to `idle`.
- Accidental disconnect / tab close → `on_client_disconnected` cancels worker;
  browser `pc.onconnectionstatechange` → `stop()`.
- **Barge-in:** Silero VAD + the aggregator pair interrupt TTS when the user speaks.
- Voice mode and text `isStreaming` send are independent; starting voice mode does
  not block typing.
- Data channel **must** be created before the offer, or Pipecat's timeout disables
  it and drops messages (enforced in the hook ordering).

## 8. Dependencies & config
- Add `pipecat-ai` (core — already includes `SmallWebRTCTransport`,
  `SmallWebRTCConnection`, `SileroVADAnalyzer`) to a new optional dependency group
  `voice`, included in `all` so `uv sync --group all` installs it. No new STT/TTS
  providers — reuses the already-present `kokoro` + transformers Whisper.
- No new required env vars. Model/backend/keys come from the same `settings` the
  text chat uses (passed in the offer body).
- Update `pyproject.toml` dependency groups + `CLAUDE.md` quick-reference if a new
  `uv sync --group voice` workflow is worth documenting.

## 9. Out of scope / risks
- Local-first, single participant (matches `SmallWebRTCTransport`); no Daily/cloud,
  no multi-user rooms.
- Cold start downloads Silero VAD and (already-present) Whisper/Kokoro models; first
  voice call has higher latency. Note in UI/README.
- `BackendLLMService` is the riskiest adapter (Pipecat `LLMService` is mixin-based);
  native `LiteLLMLLMService` fallback defined in §2.
- Heavy torch models + mic mean **E2E is manual only** (cannot run in CI).

## 10. Testing
- **Backend unit** (mock heavy services): `BackendWhisperSTTService.run_stt` yields a
  `TranscriptionFrame`; `BackendKokoroTTSService.run_tts` yields `TTSAudioRawFrame`;
  `build_voice_pipeline` constructs without a real peer; `send_app_message` is called
  with the expected event JSON (mock the connection).
- **Frontend**: hook test with mocked `getUserMedia` / `RTCPeerConnection` /
  `RTCDataChannel` asserting state transitions and that `assistant_text` messages
  update the store; light component test for the toggle + overflow menu.
- **E2E (manual)**: `uv run uvicorn main:app` + `npm run dev`; click voice toggle;
  speak; confirm caption + audio. Documented as manual in §9.

## 11. Implementation order (high level)
1. Backend: add `pipecat-ai` to `voice` dep group; `voice_pipeline.py` with the
   three custom processors + `TranscriptEmitter`; `POST /api/voice/offer` + registry
   in `app/backend/api/voice.py`; register in `main.py`.
2. Backend unit tests for processors + pipeline build + event JSON.
3. Frontend: `use-realtime-voice.ts` hook; wire toggle + captions + status pill into
   chat page; remove `Trash2`, add header overflow "Clear conversation".
4. Frontend hook/component tests; manual E2E smoke.
