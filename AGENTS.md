# AGENTS.md

> Universal agent guidance for **all AI coding harnesses** (Claude Code, Kiro, Gemini CLI,
> OpenAI Codex, Copilot Workspace, etc.) working in this repository.
> For tool-specific details see `CLAUDE.md`.  
> For coding standards and principles see `CONSTITUTION.md`.  
> For Next.js frontend rules see `nextjs-frontend/AGENTS.md`.

---

## 1. Understand the project first

This repository contains **two runtimes** that are deployed independently:

| Runtime | Root | Language | Port |
|---------|------|----------|------|
| FastAPI backend | `/` (Python) | Python 3.13 | 8000 |
| Next.js frontend | `nextjs-frontend/` | TypeScript / React 19 | 3000 |

A legacy **Streamlit** frontend (`streamlit_app.py`) still exists but is not the primary UI.  
Never mix Python backend code with Next.js frontend code. They communicate only over HTTP.

---

## 2. Before you write any code

1. **Read the relevant source files.** Never propose or apply changes to code you have not read.
2. **Check the constitution.** Verify your planned change follows `CONSTITUTION.md`.
3. **Check CI rules.** Your change must pass the same checks as `pr-checks.yml`:
   - Python: `uv run ruff check .` and `uv run ty check <paths>`
   - TypeScript: `npm run lint` inside `nextjs-frontend/`
4. **Do not invent dependencies.** Use only packages already in `pyproject.toml` (Python) or
   `nextjs-frontend/package.json` (JS/TS). If a new package is genuinely required, add it with
   the exact pinned version and explain why.

---

## 3. Mandatory checks before finishing any task

### Python changes
```bash
uv run ruff check .                  # must produce zero errors
uv run ruff format --check .         # must produce zero diffs (or run without --check to fix)
uv run ty check app/backend app/services app/core app/ingestion app/skills main.py config.py logging_config.py
uv run pytest -x -q                  # must pass
```

### TypeScript / Next.js changes
```bash
cd nextjs-frontend
npm run lint      # must produce zero errors
npm run build     # must succeed (catches type errors too)
```

### Both
- Never leave `print()` or `console.log()` debugging statements in committed code.
- Never commit secrets. Update `.env.example` for new env vars; never touch `.env`.

---

## 4. Repository layout at a glance

```
AGENTS.md               ← this file (universal agent rules)
CLAUDE.md               ← Claude-specific detail + full command reference
CONSTITUTION.md         ← coding standards and hard constraints
main.py                 ← FastAPI entry point
streamlit_app.py        ← legacy Streamlit entry point
config.py               ← global Config class
logging_config.py       ← structured logging
pyproject.toml          ← Python deps + tool config (ruff, black, mypy, ty)
version.json            ← canonical version { version, major, minor, patch }
Makefile                ← Docker lifecycle (make up/dev/prod/down/logs/health/clean)
.env.example            ← all supported environment variables with comments
app/
  backend/api/          route handlers (chat, rag, models, health)
  backend/core/         BackendConfig, embedding helpers, DEFAULT_RAG_CONFIG
  backend/models/       Pydantic models
  services/             business logic (llm_gateway, rag_service, speech, tts, …)
  skills/               pluggable skill routing (registry, base protocol)
  ingestion/            document processing pipeline
  core/                 shared utilities (environment detection, text markup)
  frontend/             legacy Streamlit UI (pages, components, services, utils)
nextjs-frontend/
  src/app/              Next.js App Router pages and layouts
  src/components/       shared React components (shadcn/ui)
  src/hooks/            custom React hooks
  src/lib/              utility functions and HTTP clients
  AGENTS.md             ← frontend-specific agent rules
tests/                  Python test suite (pytest)
scripts/                helper scripts (version.py, health-check.py, docker-setup.sh)
docs/                   extended documentation
.github/workflows/      CI/CD (pr-checks, docker-publish, release)
```

---

## 5. Environment variables

All variables are documented in `.env.example`. Key ones agents must be aware of:

| Variable | Default | Notes |
|----------|---------|-------|
| `OLLAMA_API_URL` | `http://localhost:11434` | Change to `http://host.docker.internal:11434` in Docker |
| `OPENROUTER_API_KEY` | — | Required for OpenRouter backend |
| `NVIDIA_NIM_API_KEY` | — | Required for Nvidia NIM backend |
| `SERP_API_KEY` | — | Required for web search |
| `SECRET_KEY` | placeholder | **Must be changed in production** |
| `CHROMA_DB_PATH` | `./chroma_db` | Vector store location |
| `UPLOAD_FOLDER` | `./uploads` | Uploaded document storage |
| `STORAGE_PATH` | `./storage` | RAG config and misc storage |
| `LOG_LEVEL` | `INFO` | Set `DEBUG` for verbose output |
| `DEBUG` | `0` | Set `1` to enable debug mode |

Never hard-code values that belong in environment variables.

---

## 6. Adding a new feature — decision tree

```
Is it a new LLM provider?
  → Implement in app/services/llm_gateway.py; add a case to resolve_backend_config()

Is it a new type of chat capability?
  → Add a Skill in app/skills/ implementing StreamingChatSkill protocol
  → Register it in ChatSkillRegistry (app/skills/registry.py)

Is it a new RAG capability?
  → Add a Skill implementing StreamingRAGSkill
  → Register it in RAGSkillRegistry

Is it a new document format?
  → Extend app/ingestion/file_ingest.py (format detection + extraction)

Is it a new API endpoint?
  → Add route handler in app/backend/api/
  → Add Pydantic request/response models in app/backend/models/
  → Register the router in main.py

Is it frontend UI?
  → Work in nextjs-frontend/src/ following nextjs-frontend/AGENTS.md
```

---

## 7. Testing

- Tests live in `tests/`. Mirror the `app/` structure (e.g. `tests/services/test_rag_service.py`).
- Use `pytest` with standard fixtures. No custom test runner.
- Mock external services (Ollama, SerpAPI, ChromaDB) in unit tests.
- Run `uv run pytest -x -q` before marking any Python task complete.
- Do **not** add tests unless the task explicitly asks for them — but do **not** break
  existing tests either. Always run the suite after changes.

---

## 8. Version management

Do **not** edit `version.json` by hand.

```bash
python3 scripts/version.py current          # print current version
python3 scripts/version.py bump patch       # increment patch (0.0.x)
python3 scripts/version.py bump minor       # increment minor (0.x.0)
python3 scripts/version.py bump major       # increment major (x.0.0)
python3 scripts/version.py tag              # create git tag for current version
```

The release workflow bumps the version automatically when a PR is merged to `main`.
PR labels `patch` (default), `minor`, `major` control the bump type.

---

## 9. Git hygiene

- **Never push directly to `main` or `develop`.** Always work on a feature branch.
- **Never force-push** unless explicitly instructed.
- **Never skip hooks** (`--no-verify`) unless explicitly instructed.
- Commit messages follow Conventional Commits:  
  `type(scope): short description`  
  Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `ci`, `perf`
- Stage specific files rather than `git add .` to avoid committing unintended changes.
- `.env` is git-ignored — never commit it.

---

## 10. Security

- Sanitise all file names and paths using `app/backend/api/security_utils.py`
  (`sanitize_filename`, `validate_file_size`) for any upload handling.
- Use parameterised queries. Never concatenate user input into SQL or shell commands.
- Do not log secrets, API keys, or sensitive user data.
- Do not add new CORS origins without explicit approval.
- The `SECRET_KEY` default in `.env.example` is a placeholder — remind users to rotate it.

---

## 11. What agents must NOT do

- Do not delete or move existing files without explicit instruction.
- Do not change `pyproject.toml` version field manually (use the version script).
- Do not add `print()` / `console.log()` debugging to committed code.
- Do not introduce new third-party packages without checking if an existing one covers the need.
- Do not modify `.github/workflows/` unless the task is explicitly about CI/CD.
- Do not touch Docker files unless the task is explicitly about containerisation.
- Do not hallucinate API signatures — read the source before using any internal module.
