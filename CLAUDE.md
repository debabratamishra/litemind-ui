# CLAUDE.md

> Guidance for Claude Code and other AI coding assistants working in this repository.
> See also: `AGENTS.md` (universal rules) and `CONSTITUTION.md` (coding standards).

## Quick-reference commands

```bash
# ── Python / backend ──────────────────────────────────────────────
uv sync --group all                   # install all dependency groups
uv sync --group backend               # backend only
uv sync --group frontend              # Legacy Streamlit frontend only
uv sync --group dev                   # dev tools only (ruff, black, mypy, ty, pytest)

uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload   # start backend
uv run streamlit run streamlit_app.py --server.port 8501       # start legacy Streamlit UI

uv run pytest                         # run all tests
uv run pytest tests/test_file.py      # run a single test file
uv run pytest -x -q                   # fail-fast, quiet

uv run ruff check .                   # lint (ruff)
uv run ruff format .                  # format (ruff)
uv run black .                        # alternative formatter (line-length 120)
uv run ty check app/backend app/services app/core app/ingestion app/skills main.py config.py logging_config.py
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

LiteMindUI is a **local-first AI workspace** supporting chat, RAG, web search, and voice workflows.

### Processes and ports

| Process | Entry point | Default port |
|---------|------------|-------------|
| FastAPI backend | `main.py` | 8000 |
| Next.js frontend (primary) | `nextjs-frontend/` | 3000 |
| Streamlit frontend (legacy) | `streamlit_app.py` | 8501 |

The frontend calls the backend exclusively over HTTP. They share no Python imports.

### Directory layout

```
app/
  backend/
    api/          Route handlers: chat.py, rag.py, models.py, health.py, security_utils.py
    core/         Backend-specific config, embedding helpers, BackendConfig, DEFAULT_RAG_CONFIG
    models/       Pydantic request/response models
  frontend/       Legacy Streamlit UI layer (pages/, components/, services/, utils/)
  core/           Shared utilities: environment detection, RAG formats, text markup
  services/       Backend business logic
    llm_gateway.py          Unified LLM transport (LiteLLM + Ollama direct client)
    rag_service.py          ChromaDB + BM25 hybrid retrieval, ingestion, answer composition
    rag_multi_agent.py      CrewAI multi-agent RAG orchestrator
    conversation_memory.py  Multi-turn memory with auto-summarisation
    conversation_db.py      SQLite conversation persistence
    web_search_service.py   SerpAPI REST client
    web_search_crew.py      CrewAI web-search orchestrator
    speech_service.py       STT via Whisper (transformers)
    tts_service.py          TTS via kokoro (primary) / pyttsx3 (fallback)
    host_service_manager.py Environment-aware config (Docker vs native)
    ollama.py               Direct Ollama HTTP client
  ingestion/
    file_ingest.py                     Format detection, text extraction, chunking
    enhanced_document_processor.py     PDF/DOCX/EPUB
    enhanced_extractors.py             CSV/image + EasyOCR
  skills/         Pluggable skill routing
    base.py        Protocol definitions (StreamingChatSkill, StreamingRAGSkill)
    registry.py    ChatSkillRegistry / RAGSkillRegistry
    web_search.py  Web-search chat skill
    rag.py         Standard and multi-agent RAG skills
nextjs-frontend/
  src/
    app/           Next.js 16 App Router pages and layouts
    components/    Shared React components (shadcn/ui based)
    hooks/         Custom React hooks
    lib/           Utility functions, API clients
main.py            FastAPI entry point (lifespan, route registration)
streamlit_app.py   Streamlit entry point
config.py          Global Config class (env vars, paths, performance tuning)
logging_config.py  Structured logging setup
pyproject.toml     Python dependency manifest + tool config
version.json       Canonical version { version, major, minor, patch, build_date, git_commit }
Makefile           Docker lifecycle commands
```

### Key design patterns

**LiteLLM Gateway** (`app/services/llm_gateway.py`)
Unified transport for `ollama`, `openrouter`, and `nvidia_nim`. For Ollama, uses the native `ollama` Python client directly (bypasses LiteLLM streaming) to avoid a known upstream bug. `resolve_backend_config()` normalises provider names, API bases, and keys from request params or env vars.

**Pluggable Skill Layer** (`app/skills/`)
Chat and RAG requests route through `ChatSkillRegistry` / `RAGSkillRegistry`. Each skill implements `supports()`, `validate()`, `stream()`. Add new capabilities here without touching API routes.

**RAG System** (`app/services/rag_service.py`)
ChromaDB vector store + BM25 keyword retrieval (hybrid search). Configurable embedding providers (sentence-transformers, Ollama, OpenRouter, Nvidia NIM). Documents: format detection → extraction → chunking → embedding → indexing.

**Conversation Memory** (`app/services/conversation_memory.py`)
Session-based multi-turn context. Summarises older messages when token count exceeds 75 % of the 24 K context limit. Persisted in SQLite.

**Generative UI** (`app/backend/api/chat.py`)
When `enable_generative_ui` is set, the LLM emits `` `ui:component_name` `` fenced blocks. The Next.js frontend renders these as charts, tables, metrics, progress bars, and iframe apps.

### LLM provider backends

| Backend | Key env var | Default model |
|---------|------------|---------------|
| Ollama (local) | `OLLAMA_API_URL` | `gemma3:1b` |
| OpenRouter | `OPENROUTER_API_KEY` | `meta-llama/llama-3.3-70b-instruct` |
| Nvidia NIM | `NVIDIA_NIM_API_KEY` | `meta/llama3-70b-instruct` |

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
| `Dockerfile` | Backend image |
| `Dockerfile.nextjs` | Next.js frontend image |
| `Dockerfile.streamlit` | Legacy Streamlit frontend image |
| `docker-compose.yml` | Default (local build) |
| `docker-compose.dev.yml` | Development (hot-reload) |
| `docker-compose.prod.yml` | Production |
| `docker-compose.hub.yml` | Docker Hub prebuilt images |

## Working in this repo

- Always read the relevant source file(s) before making changes.
- Run `uv run ruff check .` and `uv run ty check ...` before considering a Python change done.
- Run `npm run lint` inside `nextjs-frontend/` before considering a TypeScript change done.
- Follow the style conventions in `CONSTITUTION.md`.
- Do not modify `version.json` manually; use `python3 scripts/version.py bump`.
- Do not commit `.env` (it is in `.gitignore`); update `.env.example` instead.
- Tests live in `tests/`. Add or update tests when fixing bugs or adding features.
