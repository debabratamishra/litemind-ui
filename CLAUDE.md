# CLAUDE.md

> Guidance for Claude Code and other AI coding assistants working in this repository.
> See also: `AGENTS.md` (universal rules) and `CONSTITUTION.md` (coding standards).

## Quick-reference commands

```bash
# ── Python / backend ──────────────────────────────────────────────
uv sync --group all                   # install all dependency groups
uv sync --group backend               # backend only
uv sync --group dev                   # dev tools only (ruff, black, mypy, ty, pytest)

uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload   # start backend

uv run pytest                         # run all tests
uv run pytest tests/test_file.py      # run a single test file
uv run pytest -x -q                   # fail-fast, quiet

uv run ruff check .                   # lint (ruff)
uv run ruff format .                  # format (ruff)
uv run black .                        # alternative formatter (line-length 120)
uv run ty check backend/app/backend backend/app/services backend/app/core backend/app/ingestion backend/app/skills main.py config.py logging_config.py
uv run mypy .                         # type check

# ── Next.js frontend ──────────────────────────────────────────────
cd nextjs-frontend
npm install
npm run dev           # dev server at http://localhost:3000
npm run build         # production build
npm start             # production server
npm run lint          # eslint (Next.js core-web-vitals + TypeScript)

# ── Docker (primary workflow) ─────────────────────────────────────
make up               # default compose stack (builds images locally)
make dev              # development compose with hot-reload
make prod             # production compose
make hub-up           # pull and run prebuilt Docker Hub images
make down             # stop all services
make logs             # tail compose logs
make health           # run health-check script
make clean            # full teardown (containers + images + volumes)
make restart          # down + up

# ── Version management ────────────────────────────────────────────
python3 scripts/version.py current
python3 scripts/version.py bump patch   # or minor / major
python3 scripts/version.py tag
# version is stored in version.json; pyproject.toml is kept in sync manually
```

## Architecture overview

LiteMindUI is a **local-first AI workspace** supporting chat, RAG, web search, and realtime voice workflows.

### Processes and ports

| Process | Entry point | Default port |
|---------|------------|-------------|
| FastAPI backend | `backend/main.py` | 8000 |
| Next.js frontend (primary) | `nextjs-frontend/` | 3000 |

The frontend calls the backend exclusively over HTTP. They share no Python imports.

### Directory layout

```
backend/                FastAPI backend package
  main.py               entry point (lifespan, route registration)
  config.py             global Config class (env vars, paths, performance tuning)
  logging_config.py     structured logging setup
  app/
    backend/
      api/              route handlers: chat.py, rag.py, models.py, health.py, security_utils.py, voice.py (WebRTC SDP signaling)
        memory.py       /api/memory CRUD (user-level persistent memory)
      core/             BackendConfig, embedding helpers, DEFAULT_RAG_CONFIG
      models/           Pydantic request/response models
      user_memory_store.py  Postgres user-memory persistence (user_memories table)
    core/               shared utilities: environment detection, RAG formats, text markup
    services/           business logic (LLM, RAG, speech, voice, web search)
    ingestion/          document processing pipeline
    skills/             pluggable chat and RAG capability routing
nextjs-frontend/
  src/
    app/                Next.js 16 App Router pages and layouts
    components/         Shared React components (shadcn/ui based)
    hooks/              Custom React hooks
    lib/                Utility functions, API clients
infra/docker/           Dockerfiles and Compose workflow definitions
docs/                   documentation, design records, and the Astro site
scripts/                setup, Docker, release, and health-check helpers
tests/                  Python test suite
main.py                 compatibility shim for backend.main
config.py               compatibility shim for backend.config
logging_config.py       compatibility shim for backend.logging_config
pyproject.toml          Python dependency manifest + tool config
version.json            Canonical version { version, major, minor, patch, build_date, git_commit }
Makefile                Docker lifecycle commands
```

### Key design patterns

**LiteLLM Gateway** (`backend/app/services/llm_gateway.py`)
Unified transport for `ollama`, `openrouter`, and `nvidia_nim`. For Ollama, uses the native `ollama` Python client directly (bypasses LiteLLM streaming) to avoid a known upstream bug. `resolve_backend_config()` normalises provider names, API bases, and keys from request params or env vars.

**Pluggable Skill Layer** (`backend/app/skills/`)
Chat and RAG requests route through `ChatSkillRegistry` / `RAGSkillRegistry`. Each skill implements `supports()`, `validate()`, `stream()`. Add new capabilities here without touching API routes.

**RAG System** (`backend/app/services/rag_service.py`)
ChromaDB vector store + BM25 keyword retrieval (hybrid search). Configurable embedding providers (sentence-transformers, Ollama, OpenRouter, Nvidia NIM). Documents: format detection → extraction → chunking → embedding → indexing.

**Conversation Memory** (`backend/app/services/conversation_memory.py`)
Session-based multi-turn context. Summarises older messages when token count exceeds 75 % of the 24 K context limit. Persisted in PostgreSQL via `backend/app/backend/conversation_store.py` (`Config.DATABASE_URL`); `conversation_db.py` is legacy.

### Persistent user memory (`backend/app/backend/user_memory_store.py`, `backend/app/services/user_memory_service.py`)
Per-user durable memory in Postgres (`user_memories`, FK to `users` with ON DELETE CASCADE).
After each chat/voice exchange a fire-and-forget task asks the LLM gateway (request's own
backend/model) for JSON ops (add/update/delete, ≤3/turn) and applies them; extraction failure
never affects the response. On every chat/RAG/voice request (except multi-agent RAG mode) the backend loads the user's
memories and prepends an "About the user" system block (cap 50). Explicit "remember that…"
requests are handled by the same extraction prompt. Manual management: `/api/memory` CRUD +
Settings → Memory panel. Distinct from session-scoped `conversation_memory.py`.

**Generative UI** (`backend/app/backend/api/chat.py`)
When `enable_generative_ui` is set, the LLM emits `` `ui:component_name` `` fenced blocks. The Next.js frontend renders these as charts, tables, metrics, progress bars, and iframe apps.

### Realtime voice mode (`backend/app/backend/api/voice.py`, `backend/app/services/voice_pipeline.py`)
Browser and server establish a WebRTC peer connection. The browser POSTs an SDP
offer to `POST /api/voice/offer`; the server answers and runs a Pipecat pipeline
(`build_voice_pipeline` → `run_voice_pipeline`) as a background task. All
transcript/control events flow back over the WebRTC data channel.

- STT: `BackendWhisperSTTService` (segmented, VAD-gated). TTS: `BackendKokoroTTSService`.
- The LLM leg is `BackendLLMService` (subclasses `BaseOpenAILLMService`); `process_frame`
  drives inference, so do not call the LLM gateway directly inside the voice pipeline.
- Backend→browser events: `ready`, `user_transcript`, `assistant_text`, `assistant_end`,
  `error`, `ended`. Full contract: `docs/superpowers/specs/2026-07-18-realtime-voice-mode-design.md`.
- Voice is a **separate pipeline, not a Skill** — do not route it through the skill layer.

### LLM provider backends

| Backend | Key env var | Default model |
|---------|------------|---------------|
| Ollama (local) | `OLLAMA_API_URL` | `gemma3:1b` |
| OpenRouter | `OPENROUTER_API_KEY` | `meta-llama/llama-3.3-70b-instruct` |
| Nvidia NIM | `NVIDIA_NIM_API_KEY` | `meta/muse-glimmer-30b` |

### Key environment variables

Copy `.env.example` → `.env` and fill in secrets. Critical variables:

| Variable | Purpose |
|----------|---------|
| `OLLAMA_API_URL` | Ollama server URL (default `http://localhost:11434`) |
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `NVIDIA_NIM_API_KEY` | Nvidia NIM API key |
| `SERP_API_KEY` | SerpAPI key for web search |
| `SECRET_KEY` | Flask/FastAPI secret (change in production) |
| `CHROMA_DB_PATH` | ChromaDB storage path |
| `UPLOAD_FOLDER` | Document upload directory |
| `LOG_LEVEL` | Logging verbosity (`INFO` / `DEBUG`) |

## CI / CD

| Workflow | Trigger | What it does |
|----------|---------|-------------|
| `pr-checks.yml` | PR → `main` / `develop` | Python syntax compile, ruff lint, ty type-check |
| `docker-publish.yml` | Push to `main` / tags | Build and push backend + frontend Docker images to Docker Hub |
| `release.yml` | PR merged to `main` | Bumps `version.json`, creates git tag and GitHub release |

PRs are labelled `patch` (default), `minor`, or `major` to control the version bump.

## Document ingestion formats

PDF (PyMuPDF + pdfplumber + Camelot tables), DOCX, PPTX, XLSX, EPUB, RTF, ODF, HTML, CSV, images (EasyOCR fallback), plain text.

## Docker images

| File | Purpose |
|------|---------|
| `infra/docker/Dockerfile` | Backend image |
| `infra/docker/Dockerfile.nextjs` | Next.js frontend image |
| `infra/docker/compose/docker-compose.yml` | Default (local build) |
| `infra/docker/compose/docker-compose.dev.yml` | Development (hot-reload) |
| `infra/docker/compose/docker-compose.prod.yml` | Production |
| `infra/docker/compose/docker-compose.hub.yml` | Docker Hub prebuilt images |

## Working in this repo

- Always read the relevant source file(s) before making changes.
- Run `uv run ruff check .` and `uv run ty check ...` before considering a Python change done.
- Run `npm run lint` inside `nextjs-frontend/` before considering a TypeScript change done.
- Follow the style conventions in `CONSTITUTION.md`.
- Do not modify `version.json` manually; use `python3 scripts/version.py bump`.
- Do not commit `.env` (it is in `.gitignore`); update `.env.example` instead.
- Tests live in `tests/`. Add or update tests when fixing bugs or adding features.
