---
inclusion: always
---

# LiteMindUI — Project Overview

LiteMindUI is a **local-first AI workspace** (v0.0.27) with:
- A **FastAPI backend** (`main.py`, port 8000) — chat, RAG, web search, voice, document ingestion
- A **Next.js 16 / TypeScript frontend** (`nextjs-frontend/`, port 3000) — primary UI

The two runtimes are fully independent and communicate only over HTTP.

## Key entry points

| File | Purpose |
|------|---------|
| `main.py` | FastAPI entry point — lifespan, all route registration |
| `config.py` | Global Config class — env vars, paths, performance tuning |
| `logging_config.py` | Structured logging — always use `get_logger(__name__)` |
| `nextjs-frontend/src/app/` | Next.js App Router pages |
| `nextjs-frontend/src/components/` | Shared shadcn/ui components |
| `nextjs-frontend/src/lib/` | Utilities and HTTP client helpers |

## Core backend modules

```
backend/app/services/llm_gateway.py        → all LLM calls (Ollama, OpenRouter, Nvidia NIM)
backend/app/services/rag_service.py        → ChromaDB + BM25 hybrid RAG
backend/app/skills/registry.py             → pluggable skill routing
backend/app/ingestion/file_ingest.py       → document processing pipeline
backend/app/services/conversation_memory.py → multi-turn memory + summarisation
```

## Quick commands

```bash
# Backend
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Frontend
cd nextjs-frontend && npm run dev

# Docker (recommended)
make up       # start all services
make dev      # development mode with hot-reload
make down     # stop all services
make logs     # tail logs
make health   # health check

# Lint & type-check (Python)
uv run ruff check .
uv run ty check backend/app/backend backend/app/services backend/app/core backend/app/ingestion backend/app/skills backend/main.py backend/config.py backend/logging_config.py main.py config.py logging_config.py

# Lint & type-check (TypeScript)
cd nextjs-frontend && npm run lint && npm run build
```

## Do not

- Mix Python backend code with Next.js frontend code
- Hard-code URLs, secrets, or paths — use env vars from `.env` / `config.py`
- Edit `version.json` by hand — use `python3 scripts/version.py bump`
- Commit `.env` (git-ignored) — update `.env.example` for new variables
