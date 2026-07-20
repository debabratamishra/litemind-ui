# Design: Comprehensive Unit Test Coverage

**Date:** 2026-07-20
**Status:** Approved
**Author:** Claude (brainstorming + design)

## Goal

Bring the LiteMindUI codebase to **comprehensive per-module unit test coverage** —
meaningful tests for every module/function (happy paths, edge cases, error
handling) with **no hard coverage-percentage gate**. Scope is **both backend
(Python) and frontend (Next.js/TypeScript)**, delivered in phases.

Tests are **unit tests**: external services (Ollama, LiteLLM/OpenRouter/NIM,
ChromaDB, Whisper, Kokoro, SerpAPI, CrewAI, Pipecat) are **mocked at their
boundary**. No network calls, no model downloads, no real vector DB. The full
suite must run offline with `uv run pytest` (backend) and `npm run test`
(frontend).

## Current state (baseline)

- Backend: ~13k LOC of Python across `app/`. Only **3 test files exist**
  (`tests/test_speech_service.py`, `tests/test_voice_api.py`,
  `tests/test_voice_pipeline.py`, 370 lines) — all voice/speech focused.
- The `uv`-managed `.venv` had **no dependencies installed**; `uv run pytest`
  could not even collect the existing tests. `uv sync --group all` is run to
  install backend + dev + voice groups so all modules import.
- Frontend: 48 `.ts/.tsx` files. Only 3 tests exist
  (`message-bubble.test.tsx`, `markdown-renderer.test.tsx`,
  `use-realtime-voice.test.ts`). **No `vitest.config.*` exists**, yet tests use
  the `@/` path alias → they currently cannot resolve. A config must be added.
- pytest is configured with `anyio_mode = "auto"` (async tests via anyio).

## Approach (recommended)

**Mirrored test layout + boundary mocking + Vitest/jsdom.**

1. `tests/` mirrors `app/` so every source module has a clear test home.
2. A shared `tests/conftest.py` provides common fixtures (temp dirs, fake
   `Config`, monkeypatch helpers, HTTP mocking).
3. External services are mocked at their boundary (the function/class that
   crosses into network/model inference), never by stubbing internals.
4. Frontend uses Vitest + React Testing Library + jsdom with a proper config
   and the `@/` alias.

### Alternatives considered

- *Centralized large test files per subsystem* — simpler to navigate but produces
  unwieldy files and harder review. Rejected in favor of mirrored layout.
- *Stub heavy third-party imports via `sys.modules`* to avoid installing torch/
  chromadb/kokoro/pipecat — fastest install, but fragile and cannot faithfully
  exercise those libraries. Rejected; user chose full deps + boundary mocking.
- *Hard `fail_under` coverage gate in CI* — rejected per the "comprehensive
  per-module" (not "line % threshold") choice; coverage is reported, not gated.

## Backend test architecture

### Conventions

- Files: `tests/<mirrored-path>/test_<module>.py`, mirroring `app/...`.
- Run with `uv run pytest` (root). `anyio_mode = "auto"` covers `async def test_*`.
- Mocks via `unittest.mock` (`MagicMock`/`AsyncMock`) and `monkeypatch`.
- HTTP mocking via `httpx.MockTransport` / monkeypatched `httpx.AsyncClient`
  (preferred — no new dependency). Only use `respx` if it is already installed by
  `uv sync --group all`; do not add it as a new dependency.
- Use FastAPI `TestClient` (sync wrapper) or `httpx.AsyncClient` +
  `ASGITransport` for API routes, with skill/LLM layers monkeypatched.
- Each test is offline by construction; no real model loads or network.

### Shared `tests/conftest.py` fixtures (proposed)

- `tmp_upload_dir` / `tmp_chroma_dir` — temp dirs bound to `UPLOAD_FOLDER`,
  `CHROMA_DB_PATH` via monkeypatch.
- `fake_config` — a `Config` (or lightweight stand-in) with deterministic values.
- `mock_ollama` — monkeypatched Ollama client returning canned chat/embed responses.
- `mock_llm_gateway` — monkeypatched LiteLLM completion/stream returning canned text.
- `mock_chromadb` — monkeypatched `chromadb.Client`/`HttpClient` + collection
  (add/get/query/upsert return canned data).
- `mock_serpapi` — monkeypatched outbound `httpx` for web search.
- `mock_kokoro` / `mock_whisper` — monkeypatched model load + inference entrypoints.

## Backend module coverage plan (phased)

Ordered low-dependency → high-value logic → externals → routes → ingestion.

### Phase B1 — Pure logic / low dependency

| Module | What to test |
|--------|--------------|
| `app/core/text_markup.py` | `_match_tag`, `_find_tagged_spans`, `extract_tagged_sections`, `remove_tagged_sections`, `replace_fenced_code_blocks` — nested tags, unclosed tags, empty content, multiple fences, no fences. |
| `app/core/rag_formats.py` | Format detection helpers, formatting of retrieved contexts, edge cases. |
| `app/core/environment.py` | Environment detection (Docker vs native), path resolution, boolean parsing. |
| `app/skills/base.py` | `StreamingChatSkill` / `StreamingRAGSkill` protocol defaults (`supports`/`validate`/`stream`), `name`. |
| `app/skills/registry.py` | `ChatSkillRegistry.resolve` (first-match, none), `RAGSkillRegistry.get`/`resolve`, immutability of `skills`. |
| `app/skills/rag.py` | RAG skill `supports`/`validate`/`stream` branch logic (standard vs multi-agent). |
| `app/skills/web_search.py` | Web-search skill `supports`/`validate`/`stream` routing. |
| `app/backend/models/api_models.py` | Pydantic request/response models: validation, defaults, serialization, required vs optional fields. |
| `app/backend/core/config.py` | `BackendConfig`, `DEFAULT_RAG_CONFIG` defaults, env overrides, merge behavior. |
| `app/backend/core/embeddings.py` | Embedding provider selection/normalization; mock the actual model so no download. |
| `app/backend/core/ollama_models.py` | Model listing/parsing helpers; mock Ollama calls. |
| `app/backend/api/security_utils.py` | Auth/signature/secret helpers — hashing, comparison, error on bad input. |
| `app/backend/api/health.py` | Health endpoint payloads (healthy/degraded), dependency checks. |
| `app/backend/api/models.py` | Models endpoint: listing, filtering, error on provider failure. |

### Phase B2 — Service layer (mocked externals)

| Module | What to test (mocked boundary) |
|--------|-------------------------------|
| `app/services/llm_gateway.py` | `resolve_backend_config` normalization (ollama/openrouter/nvidia_nim); completion + streaming for each provider; Ollama direct-client streaming path; error/retry/timeout handling. |
| `app/services/ollama.py` | Direct Ollama HTTP client: chat, embeddings, model list; error handling; mock `httpx`. |
| `app/services/conversation_db.py` | SQLite CRUD: create/append/load/delete sessions; message ordering; persistence across reopen; corrupt-DB handling. |
| `app/services/conversation_memory.py` | Multi-turn context assembly; summarization trigger when token budget exceeds 75% of 24K limit; truncation; empty history. |
| `app/services/host_service_manager.py` | Environment-aware config selection (Docker vs native); port/host resolution. |
| `app/services/web_search_service.py` | SerpAPI REST client: request construction, response parsing, result normalization, rate-limit/error handling (mock `httpx`). |
| `app/services/web_search_crew.py` | CrewAI orchestrator wiring; mock crew `.kickoff`/run; result shaping. |
| `app/services/tts_text_processing.py` | Markdown/URL cleanup before synthesis; edge cases (no markup, only URLs, nested). **Mostly pure — strong coverage target.** |
| `app/services/tts_service.py` | Kokoro primary + pyttsx3 fallback selection; voice selection; error fallback path; mock model load + synthesis. |
| `app/services/speech_service.py` | Whisper STT pipeline: transcription entrypoint, audio preprocessing, error handling; mock transformers pipeline (refresh existing `test_speech_service.py`). |
| `app/services/rag_multi_agent.py` | Multi-agent RAG orchestrator: agent assembly, routing, result merge; mock crew/LLM. |
| `app/services/rag_service.py` | Hybrid retrieval (ChromaDB vector + BM25 keyword), ingestion pipeline, chunking, embedding/index, answer composition, query error handling; mock ChromaDB + embedder. **Largest module — prioritize.** |

### Phase B3 — API routes

| Module | What to test (mocked layers) |
|--------|------------------------------|
| `app/backend/api/chat.py` | Chat + streaming chat endpoints: skill resolution, generative-UI fenced-block emission, SSE/stream framing, error responses, param validation. |
| `app/backend/api/rag.py` | RAG query + ingestion endpoints: request handling, skill routing, error paths. |
| `app/backend/api/voice.py` | WebRTC SDP offer endpoint: returns answer, errors on bad offer; mock pipeline builder/runner (refresh existing `test_voice_api.py` / `test_voice_pipeline.py` to run under new env). |

### Phase B4 — Ingestion

| Module | What to test (mocked heavy libs) |
|--------|----------------------------------|
| `app/ingestion/file_ingest.py` | Format detection by extension/MIME, text extraction dispatch, chunking strategy + sizes/overlap, unsupported-format errors. |
| `app/ingestion/enhanced_extractors.py` | CSV/image (EasyOCR) extractor paths; mock `easyocr`, `pandas`; missing-file/empty-file handling. |
| `app/ingestion/enhanced_document_processor.py` | PDF/DOCX/EPUB processing: mock `PyMuPDF`/`pdfplumber`/`camelot`/`python-docx`/`openpyxl`; table extraction, metadata, error on corrupt input. |

## Frontend test architecture

### Setup to add

- `nextjs-frontend/vitest.config.ts`: `environment: "jsdom"`, `resolve.alias` mapping
  `@` → `./src`, a `setupFiles` entry (e.g. `./vitest.setup.ts`), and globals on.
- `nextjs-frontend/vitest.setup.ts`: import `@testing-library/jest-dom` matchers
  if added; basic mocks for `matchMedia`/ResizeObserver (needed by Radix/next-themes).
- Optional: `@vitest/coverage-v8` devDependency for coverage reporting (not gated).

### Suites (mirror existing `.test.ts(x)` style)

| Area | Files to cover |
|------|----------------|
| `src/lib/` | `utils.ts` (cn/format helpers), `types.ts` (type guards/validation), `api.ts` (fetch wrapping, error mapping — mock `fetch`), `backend-proxy.ts`, `generative-ui.ts` (payload parsing/dispatch), `store.ts` (Zustand store actions/state), `web-search-citations.ts` (citation parsing/numbering). |
| `src/hooks/` | `use-rag-upload.ts` (upload flow, progress, error), `use-voice-input.ts` (dictation toggle, transcript callback), `use-realtime-voice.ts` (WebRTC lifecycle, event handling — extend existing test). |
| `src/components/` | `settings-panel.tsx`, `knowledge-base.tsx`, `citations.tsx`, `generative-ui-renderer.tsx`, `rag-attach-button.tsx`, `sidebar.tsx`, `theme-toggle.tsx`, `theme-provider.tsx`, `voice-activity.tsx`, `voice-input-button.tsx`; logic-bearing `ui/` primitives (e.g. `progress`, `slider`, `tabs`, `select` interaction). |
| Already covered (extend) | `message-bubble.test.tsx`, `markdown-renderer.test.tsx`, `use-realtime-voice.test.ts`. |

React Testing Library for rendering; `vi.fn()`/`vi.mock()` for API/store boundaries;
match the mocking style of the existing tests.

## Coverage reporting & CI

- Backend: `uv run pytest --cov=app --cov-report=term-missing` (pytest-cov already in
  `dev` group). Add `addopts`/coverage config to `[tool.pytest.ini_options]`; omit
  `__init__.py`. No `fail_under` gate.
- Frontend: optional `vitest --coverage` after adding coverage dep.
- README / `make` note documenting `uv run pytest` and `npm run test`.
- Final summary reports per-module coverage numbers so gaps are visible; **no hard
  CI gate** (per "comprehensive per-module" choice).

## Definition of done

- `uv run pytest` passes end-to-end, offline, with meaningful tests for every
  backend module listed above.
- `npm run test` (Vitest) passes end-to-end for every frontend file listed above.
- Each test file is focused and runnable in isolation.
- Shared fixtures/conftest avoid duplication; no real network/model/DB calls.

## Out of scope

- Integration/e2e tests against real Ollama/ChromaDB/LLM providers.
- Performance/benchmark tests.
- Changing application behavior — tests only; if a bug is found, report it and
  add a regression test, but do not silently "fix" unrelated code.

## Assumptions

- `uv sync --group all` succeeds in this environment (installing torch CPU,
  chromadb, kokoro, pipecat, etc.). If a heavy dep fails to install, that
  module's tests will mock its import boundary so the rest still runs.
- The frontend packages are already installed (`npm install` run previously);
  if not, run it before `npm run test`.
