# CONSTITUTION.md

> Hard coding standards and architectural principles for **LiteMindUI**.  
> These rules apply to every agent and every human contributor.  
> Rules here take precedence over personal preference.  
> See `AGENTS.md` for workflow rules and `CLAUDE.md` for commands.

---

## 1. Core principles

1. **Correctness first.** A working, correct change beats a clever but fragile one.
2. **Read before you write.** Never modify a file you have not read in this session.
3. **Minimal diff.** Change only what the task requires. Do not refactor unrelated code.
4. **Explicit over implicit.** Avoid magic; prefer clear, traceable data flow.
5. **Fail loudly.** Raise descriptive exceptions; never swallow errors silently.
6. **Local-first.** Default behaviour must work without external API keys (Ollama local only).

---

## 2. Python standards

### 2.1 Language and toolchain
- Python **3.13** only (`requires-python = ">=3.13,<3.14"` in `pyproject.toml`).
- Package manager: **uv**. Do not use pip, poetry, or pipenv to install or manage deps.
- Formatter: **ruff format** (line length 120). Run before every commit.
- Linter: **ruff check** (line length 120). Zero warnings allowed.
- Type checker: **ty** (Astral). Zero errors allowed on checked paths.
- Secondary type checker: **mypy** for IDE integration.

### 2.2 Style
- Line length: **120 characters**.
- Imports: stdlib → third-party → local, each group alphabetically sorted (ruff handles this).
- No wildcard imports (`from module import *`).
- Use `from __future__ import annotations` only when needed for forward references.
- f-strings preferred over `.format()` or `%` formatting.
- Single quotes for strings unless the string contains a single quote.

### 2.3 Type annotations
- **All** function signatures must have type annotations (parameters + return type).
- Use `from typing import ...` for `Optional`, `Union`, `Any`, `List`, `Dict` only on Python < 3.10
  syntax; for Python 3.13 prefer built-in generics (`list[str]`, `dict[str, int]`, `X | Y`).
- `Any` is a last resort — justify it with a comment.
- Pydantic models must annotate every field.

### 2.4 Logging
- Use the project logger from `logging_config.py`:
  ```python
  from logging_config import get_logger
  logger = get_logger(__name__)
  ```
- Do not use `print()` for diagnostic output in production paths.
- Log levels: `DEBUG` for verbose traces, `INFO` for lifecycle events, `WARNING` for recoverable
  issues, `ERROR` for caught exceptions, `CRITICAL` for startup failures.
- Never log secrets, API keys, or raw user PII.

### 2.5 Error handling
- Catch specific exceptions, not bare `except Exception` unless re-raising or at a top-level
  boundary (e.g. FastAPI exception handlers).
- FastAPI route handlers must return structured `JSONResponse` or raise `HTTPException` —
  never let unhandled exceptions propagate to the ASGI layer.
- Include contextual information in exception messages (what was attempted, what failed).

### 2.6 Async
- FastAPI route handlers and service methods that perform I/O must be `async def`.
- Do not call blocking code (file I/O, `subprocess`, heavy CPU work) inside `async def` without
  `asyncio.to_thread()` or a process-pool executor.
- Prefer `httpx.AsyncClient` over `requests` in async contexts.

### 2.7 Configuration
- All runtime configuration goes through `config.py` (`Config` class) or
  `app/backend/core/config.py` (`BackendConfig`).
- Never hard-code URLs, file paths, secrets, or numeric thresholds in business logic.
- New environment variables must be added to `.env.example` with a description comment.

### 2.8 Pydantic models
- Request and response models live in `app/backend/models/`.
- Use `Optional[X] = None` (or `X | None = None`) for truly optional fields.
- Validate and sanitise at the model boundary — do not re-validate downstream.

### 2.9 Security
- All uploaded file names must pass through `sanitize_filename()` from
  `app/backend/api/security_utils.py`.
- All uploaded file sizes must be validated with `validate_file_size()`.
- Use parameterised queries for all database operations (SQLAlchemy ORM or `?` placeholders).
- CORS origins are configured in `main.py`; do not widen them without approval.

---

## 3. TypeScript / Next.js standards

### 3.1 Toolchain
- Node version: whatever `.nvmrc` or `package.json` `engines` specifies (currently Next.js 16,
  React 19, TypeScript 5).
- Package manager: **npm** (lock file is `package-lock.json`). Do not use yarn or pnpm.
- Linter: **ESLint** with `eslint-config-next/core-web-vitals` + `eslint-config-next/typescript`.
  Zero errors allowed (`npm run lint`).
- Type checker: **TypeScript strict mode** (`"strict": true` in `tsconfig.json`).
  Build must succeed (`npm run build`) — it catches type errors.
- Formatter: not enforced by CI but prefer Prettier defaults (2-space indent, single quotes,
  trailing commas in multi-line).

### 3.2 Style
- `"strict": true` TypeScript — no implicit `any`.
- Functional components only. No class components.
- Custom hooks in `src/hooks/`. Prefix with `use`.
- Shared utilities in `src/lib/`. No business logic in components.
- Components in `src/components/`. Use shadcn/ui primitives where available.
- Use absolute imports via the `@/*` alias (maps to `src/`).
- No `// @ts-ignore` or `// @ts-expect-error` without an explanatory comment.

### 3.3 State management
- Local UI state: `useState` / `useReducer`.
- Cross-component / persistent client state: **Zustand** (already a dependency).
- Do not introduce Redux, MobX, Jotai, or other state libraries.

### 3.4 Data fetching
- Use `fetch` with the Next.js App Router patterns (server components, `use server`, etc.).
- For client-side streaming (chat, RAG): consume SSE / `ReadableStream` directly.
- Do not add React Query, SWR, or Axios unless explicitly approved.

### 3.5 Styling
- **Tailwind CSS v4** + **shadcn/ui** components.
- No inline `style={{}}` objects except for dynamic values that cannot be expressed as Tailwind.
- No plain CSS files unless absolutely necessary; prefer Tailwind utility classes.
- Dark/light theme via `next-themes` — always test both modes for new UI.

### 3.6 Accessibility
- Interactive elements must have accessible labels (`aria-label`, `aria-labelledby`, or visible text).
- Avoid `tabIndex > 0`.
- Colour contrast must meet WCAG 2.1 AA (4.5:1 for normal text, 3:1 for large text).
- All images need meaningful `alt` text or `alt=""` for decorative images.

### 3.7 Performance
- Lazy-load large components with `dynamic(() => import(...))`.
- Do not import the entire icon library — import individual icons from `lucide-react`.
- Avoid large synchronous computations in render paths.

---

## 4. API contract

- The backend exposes a REST API documented at `http://localhost:8000/docs` (Swagger UI).
- The frontend must not assume any backend internals — call only documented endpoints.
- Breaking changes to the API (renamed fields, removed endpoints, changed response shapes)
  require updating both the backend route and the frontend client code in the same PR.
- Streaming endpoints use **Server-Sent Events (SSE)** or plain chunked transfer encoding.
  The frontend handles both.

---

## 5. Architecture constraints

### 5.1 Skill layer
- New chat or RAG capabilities **must** be implemented as Skills (`app/skills/`).
- Never add conditional capability logic directly inside route handlers.
- Skills are resolved by the registry in priority order; the first `supports()` match wins.

### 5.2 LLM gateway
- All LLM calls go through `app/services/llm_gateway.py`.
- Never call Ollama, OpenRouter, or Nvidia NIM APIs directly from route handlers or Skills.
- New providers require a new case in `resolve_backend_config()` and corresponding streaming logic.

### 5.3 RAG system
- Chunking parameters (size, overlap) are user-configurable via `RAGConfigRequest`.
- Do not hard-code chunk sizes in business logic.
- Hybrid search (vector + BM25) is on by default; do not remove it.

### 5.4 Document ingestion
- All document processing goes through `app/ingestion/file_ingest.py`.
- Format detection must be by MIME type or magic bytes, not file extension alone.
- OCR (EasyOCR) is an optional fallback for images — it is slow; do not call it eagerly.

### 5.5 Conversation memory
- Memory is session-scoped and lives in `app/services/conversation_memory.py`.
- Summarisation kicks in at 75 % of the 24 K context limit — do not change this threshold
  without load-testing the summarisation path.

### 5.6 Environment detection
- Container vs native detection is centralised in `app/core/environment.py` (singleton
  `EnvironmentDetector`). Do not duplicate this logic elsewhere.

---

## 6. File and naming conventions

| Artifact | Convention | Example |
|----------|-----------|---------|
| Python modules | `snake_case.py` | `rag_service.py` |
| Python classes | `PascalCase` | `RAGService` |
| Python functions/methods | `snake_case` | `retrieve_documents` |
| Python constants | `SCREAMING_SNAKE_CASE` | `DEFAULT_RAG_CONFIG` |
| TypeScript files | `kebab-case.ts(x)` | `chat-service.ts` |
| React components | `PascalCase.tsx` | `ChatMessage.tsx` |
| React hooks | `useCamelCase.ts` | `useStreamingChat.ts` |
| CSS class names | Tailwind utilities only | — |
| Env var names | `SCREAMING_SNAKE_CASE` | `OLLAMA_API_URL` |
| Test files (Python) | `test_<module>.py` | `test_rag_service.py` |

---

## 7. Dependency policy

- **Prefer existing dependencies** over adding new ones.
- Before adding a package, verify no existing dependency already covers the need.
- Pin new Python packages to **exact versions** in `pyproject.toml` optional groups.
- Pin new JS packages to **exact versions** in `package.json` (no `^` or `~` for new deps).
- Heavy ML packages (PyTorch, transformers, sentence-transformers) go in the `backend` group only —
  never in `frontend` or `dev`.
- Do not add packages that are only available for Python < 3.13 or incompatible with Python 3.13.

---

## 8. Documentation

- Every new public Python function or class must have a **docstring** (one-line summary minimum).
- Complex logic must have inline comments explaining *why*, not just *what*.
- New environment variables must be documented in `.env.example`.
- New API endpoints must have a `summary` and `description` in the FastAPI route decorator.
- Do not add or modify `README.md` unless the task is explicitly documentation.

---

## 9. What is permanently off-limits

- Storing secrets in source code, config files, or committed `.env` files.
- Disabling ruff, ty, or mypy on entire files with `# noqa: all` or `# type: ignore` blanket comments.
- Synchronous blocking calls inside `async def` handlers without executor offloading.
- Returning raw Python exceptions or stack traces in HTTP responses (use `HTTPException`).
- Adding client-side environment variables prefixed `NEXT_PUBLIC_` that expose secrets.
- Using `eval()`, `exec()`, or `pickle` on user-supplied data.
- Shipping dead code, commented-out code blocks, or TODO stubs in production paths.
