# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install all dependencies (backend + frontend + dev)
uv sync --group all

# Install only specific groups
uv sync --group backend
uv sync --group frontend
uv sync --group dev

# Start backend (FastAPI)
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Start frontend (Streamlit)
uv run streamlit run streamlit_app.py --server.port 8501

# Run tests
uv run pytest

# Run a single test file
uv run pytest tests/test_file.py

# Lint
uv run ruff check .

# Format
uv run black .

# Type check
uv run mypy .

# Docker (primary development workflow)
make up          # default compose stack
make dev         # development compose stack
make prod        # production compose stack
make logs        # tail container logs
make health      # health check
make down        # stop services
make clean       # full cleanup

# Version management
# python3 scripts/version.py bump patch|minor|major
# python3 scripts/version.py current
# python3 scripts/version.py tag
```

## Architecture

LiteMindUI is a **local-first AI workspace** with a **FastAPI backend** and a **Streamlit frontend** for chat, RAG, web search, and voice-enabled workflows.

### Two-Process Architecture

The app runs as two separate processes that communicate over HTTP:

1. **FastAPI backend** (`main.py`) — serves API endpoints on port 8000
2. **Streamlit frontend** (`streamlit_app.py`) — serves UI on port 8501, calls backend via HTTP

### Directory Layout

```
app/
  backend/          FastAPI routes, Pydantic schemas, and backend config
    api/            Route handlers (chat.py, rag.py, models.py, health.py, security_utils.py)
    core/           Backend-specific config (embeddings, model helpers, BackendConfig)
    models/         Pydantic request/response models
  frontend/         Streamlit UI layer
    pages/          Page-level components (chat_page.py, rag_page.py)
    components/     Reusable UI components (voice_input, tts, generative_ui, sidebar, etc.)
    services/       HTTP client for backend REST API (backend_service, chat_service, rag_service)
    config.py       Frontend config (URLs, timeouts, system prompts)
    utils/          Audio recording, memory management, text processing
  core/             Shared utilities (environment detection, RAG formats, text markup)
  services/         Backend service layer (business logic)
    llm_gateway.py  Unified LLM transport via LiteLLM (Ollama, OpenRouter, Nvidia NIM)
    rag_service.py  RAG: ChromaDB + BM25 hybrid retrieval, document ingestion, answer composition
    rag_multi_agent.py  CrewAI multi-agent RAG orchestrator
    conversation_memory.py  Multi-turn conversation memory with summarization
    conversation_db.py  SQLite conversation persistence
    web_search_service.py  SerpAPI web search client
    web_search_crew.py  CrewAI web search orchestrator
    speech_service.py  Speech-to-text (Whisper via transformers)
    tts_service.py  Text-to-speech (kokoro primary, pyttsx3 fallback)
    host_service_manager.py  Docker/native environment detection, service connectivity
    ollama.py       Direct Ollama HTTP client
  ingestion/        Document processing pipeline
    file_ingest.py  Format detection, text extraction, chunking
    enhanced_document_processor.py  PDF/DOCX/EPUB enhanced extraction
    enhanced_extractors.py  CSV/image enhanced extraction with OCR
  skills/           Pluggable skill layer for routing requests
    base.py         Protocol definitions (StreamingChatSkill, StreamingRAGSkill)
    registry.py     ChatSkillRegistry and RAGSkillRegistry for skill resolution
    web_search.py   Web search chat skill
    rag.py          Standard and multi-agent RAG skills
main.py             FastAPI entrypoint (lifespan, route registration, startup)
streamlit_app.py    Streamlit entrypoint (page config, session state, routing)
config.py           Global config (paths, URLs, performance tuning)
logging_config.py   Structured logging setup
```

### Key Design Patterns

**LiteLLM Gateway** (`app/services/llm_gateway.py`): A unified transport layer that abstracts LLM providers. Supports three backends: `ollama`, `openrouter`, and `nvidia_nim`. For Ollama, it bypasses LiteLLM's streaming and uses the `ollama` Python client directly to avoid a LiteLLM streaming bug. The `resolve_backend_config()` function normalizes backend names, API bases, and API keys from request parameters or environment variables.

**Pluggable Skill Layer** (`app/skills/`): Chat and RAG requests are routed through registries (`ChatSkillRegistry`, `RAGSkillRegistry`) that resolve the first matching skill. Each skill implements a protocol (`StreamingChatSkill`, `StreamingRAGSkill`) with `supports()`, `validate()`, and `stream()` methods. This allows adding new capabilities (web search, multi-agent RAG, etc.) without modifying the API routes.

**RAG System** (`app/services/rag_service.py`): Uses ChromaDB for vector storage with configurable embedding providers (HuggingFace sentence-transformers, Ollama, OpenRouter, Nvidia NIM). Hybrid search combines vector similarity (ChromaDB) with keyword retrieval (BM25). Documents go through a pipeline: format detection → text extraction → chunking → embedding → indexing.

**Conversation Memory** (`app/services/conversation_memory.py`): Manages multi-turn conversation context with automatic summarization. When token count exceeds 75% of the 24K context limit, older messages are summarized. Supports session-based memory with SQLite persistence.

**Environment Detection** (`app/core/environment.py`): Singleton `EnvironmentDetector` that detects Docker vs native execution using multiple heuristics (`.dockerenv` file, cgroup info, environment variables, mount points). `HostServiceManager` (`app/services/host_service_manager.py`) builds on this to provide environment-aware configuration (Ollama URLs, cache directories, paths).

**Generative UI** (`app/backend/api/chat.py`): When `enable_generative_ui` is set, the system prompt instructs the LLM to emit rich UI components using ````ui:component_name` fenced code blocks. The Streamlit frontend renders these as rich components (data tables, charts, metrics, webapps, iframe apps, etc.).

### Key Configuration

- `.env` / `.env.example` — all environment variables
- `config.py` — `Config` class loads env vars, detects container environment, sets paths and performance tuning
- `app/backend/core/config.py` — `BackendConfig` manages RAG config persistence and dynamic settings
- `app/frontend/config.py` — frontend URLs, timeouts, system prompts

### Document Processing

The ingestion pipeline (`app/ingestion/`) supports: PDF (PyMuPDF + pdfplumber + Camelot tables), DOCX, PPTX, XLSX, EPUB, RTF, ODF, HTML, CSV, images (with EasyOCR fallback), and plain text. Chunking is configurable by size. Image indexing within documents is optional (`ENABLE_SIMPLE_IMAGE_INDEXING`).

### LLM Provider Backends

| Backend | Config | Default Model |
|---------|--------|---------------|
| Ollama | `OLLAMA_API_URL` | `gemma3:1b` |
| OpenRouter | `OPENROUTER_API_KEY` | `openai/gpt-4o-mini` |
| Nvidia NIM | `NVIDIA_NIM_API_KEY` | `meta/llama3-70b-instruct` |

### Docker Deployments

- `Dockerfile` — backend image
- `Dockerfile.streamlit` — frontend image
- `docker-compose.yml` — default (local build)
- `docker-compose.dev.yml` — development (hot reload)
- `docker-compose.prod.yml` — production
- `docker-compose.hub.yml` — prebuilt images from Docker Hub
- `install.sh` — quick installer pulling Docker Hub images
- `scripts/` — health checks, startup validation, cache setup, Docker setup helpers