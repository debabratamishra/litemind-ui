# Self-Learning Harness Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strip auth, extract the LLM-provider-neutral kernel with encrypted key vault + UI, and cut the Docker image to a core profile.

**Architecture:** One FastAPI process. `backend/app/kernel/` holds provider routing and the Postgres-backed Fernet key vault; plugins/sub-agents come in Phase 2/3. Auth files deleted; all routes resolve a seeded system user. uv dependency groups split the image into core/ml/voice/ocr profiles; heavy imports go lazy so a core-only image boots.

**Tech Stack:** Python 3.13, FastAPI, asyncpg, cryptography (Fernet), uv, Next.js 16 App Router, Docker multi-stage.

**Spec:** `docs/superpowers/specs/2026-11-15-self-learning-harness-design.md`

## Global Constraints

- `uv run` wrappers for every Python command; `uv sync --group all` only.
- Branch `feat/self-learning-harness`; multi-line commit messages; no co-authored-by lines.
- Tests offline: providers mocked at boundary, matching `tests/conftest.py` conventions.
- Single local user: `SYSTEM_USER_ID = "00000000-0000-0000-0000-000000000001"`.
- Resolution order everywhere: explicit request param > key vault > env var.
- Keys never echoed: API responses carry `configured: bool` and metadata only.

## Review Focus

- Fernet key lost on redeploy → stored credentials undecryptable. Pinned by T3 key-file roundtrip test.
- Orphaned conversations break list API after auth removal. Pinned by T2 backfill test.
- A route still 401s without a token. Pinned by T1 route test.
- Core-only image crashes importing chromadb/torch at module level. Pinned by T6 AST test + import gate.
- Keys API leaks plaintext/token. Pinned by T5 response-shape test.

---

### Task 1: Delete auth, swap to system user

**Files:**
- Delete: `backend/app/backend/api/auth.py`, `auth_deps.py`, `auth_service.py`, `auth_verify.py`
- Delete: `tests/test_auth_deps.py`, `test_auth_router.py`, `test_auth_service.py`, `test_auth_verify.py`, `test_chat_auth.py`, `test_rag_voice_auth.py` (read `test_rag_voice_auth.py` first; move any non-auth cases into `tests/test_rag_voice.py`)
- Create: `backend/app/backend/api/deps.py`
- Modify: `backend/main.py:23,281` (drop import + `include_router`), `backend/app/backend/api/{conversations,memory,chat,rag,voice}.py` (swap dependency)
- Test: `tests/test_auth_removal.py`

**Interfaces:**
- Produces: `deps.User` (pydantic `id: str`, `email: Optional[str] = None`, `name: Optional[str] = None`), `deps.SYSTEM_USER_ID`, `deps.get_system_user() -> User`.
- Consumes: nothing.

- [ ] **Step 1:** Write failing test — `GET /api/conversations` with no Authorization header/cookie returns non-401 (use the app/client pattern from `tests/backend/api/`).
- [ ] **Step 2:** Run `uv run pytest tests/test_auth_removal.py -v`; expect FAIL (401).
- [ ] **Step 3:** Create `deps.py` with `User`, `SYSTEM_USER_ID`, `get_system_user()`; delete the four auth API files and six test files; replace every `Depends(get_current_user)` with `Depends(get_system_user)` and `User` imports with `deps.User`; drop `auth_api` from `main.py`.
- [ ] **Step 4:** Run test; expect PASS. Also `uv run pytest tests -q` (full suite, auth tests gone).
- [ ] **Step 5:** Commit: `refactor: drop gotrue auth, routes use seeded system user`

### Task 2: Seed system user + backfill orphans

**Files:**
- Modify: `backend/app/backend/conversation_store.py` (`init_schema`, line ~123)
- Test: append to `tests/test_conversation_store.py`

**Interfaces:**
- Consumes: `deps.SYSTEM_USER_ID` (T1).
- Produces: `init_schema()` that idempotently inserts the system user and repoints orphaned `conversations.user_id` (and `messages.user_id`, if that column exists — check schema at line ~47).

- [ ] **Step 1:** Write failing test — fresh store, `init_schema()`, assert `SELECT 1 FROM users WHERE id='00000000-0000-0000-0000-000000000001'`; insert conversation with random uuid user before seed, assert backfill to system user.
- [ ] **Step 2:** Run `uv run pytest tests/test_conversation_store.py -v`; expect FAIL.
- [ ] **Step 3:** In `init_schema`, after CREATE TABLEs: `INSERT INTO users (id) VALUES ($1) ON CONFLICT DO NOTHING`; then `UPDATE conversations SET user_id=$1 WHERE user_id NOT IN (SELECT id FROM users)`; same for messages if column exists. Update the "source of truth is GoTrue" comment at line ~10.
- [ ] **Step 4:** Run test; PASS.
- [ ] **Step 5:** Commit: `fix(store): seed system user and backfill orphaned rows on boot`

### Task 3: Encrypted key vault

**Files:**
- Create: `backend/app/kernel/__init__.py`, `backend/app/kernel/keyvault.py`
- Test: `tests/backend/test_keyvault.py`

**Interfaces:**
- Consumes: `Config` storage path + `DATABASE_URL` (mirror `user_memory_store.py` singleton/pool pattern).
- Produces: `KeyVault` with `init_schema()`, `set_credential(kind, provider, label, plaintext) -> str` (row id), `get_credential(kind, provider) -> Optional[str]`, `list_credentials() -> list[dict]` (id/kind/label/provider/updated_at only), `delete_credential(id) -> bool`; module fn `get_key_vault() -> KeyVault`. Table `user_credentials(id UUID PK, kind TEXT, label TEXT, provider TEXT, token TEXT, updated_at TIMESTAMPTZ)`.

- [ ] **Step 1:** Write failing tests — set/get roundtrip, `list_credentials()` has no token field, delete removes, unknown → `None`, key file persisted at `<storage>/keyvault.key` mode 0600 and reused by a second `KeyVault` instance (decrypt still works).
- [ ] **Step 2:** Run `uv run pytest tests/backend/test_keyvault.py -v`; expect FAIL (no module).
- [ ] **Step 3:** Implement `keyvault.py`: Fernet key from env `KEYVAULT_KEY`, else generate + write 0600 file under storage dir; `cryptography.fernet.Fernet` for token; `add "cryptography"` to pyproject core group.
- [ ] **Step 4:** Run tests; PASS.
- [ ] **Step 5:** Commit: `feat(kernel): encrypted credential vault with persisted fernet key`

### Task 4: Kernel provider router + providers API

**Files:**
- Create: `backend/app/kernel/router.py`, `backend/app/backend/api/kernel.py`
- Modify: `backend/app/services/llm_gateway.py` (thin re-export shim), `backend/main.py` (include router)
- Test: `tests/backend/test_kernel_router.py`

**Interfaces:**
- Consumes: `KeyVault` (T3), `llm_gateway` resolution logic (`_resolve_api_base`, `_resolve_api_key`, `resolve_backend_config` at lines 340-365).
- Produces: `router.resolve_provider_config(capability: str, *, api_key=None, api_base=None, model=None) -> BackendConfig`; `CAPABILITIES = ("llm", "vlm", "stt", "tts", "embeddings", "search")` with env fallback map; API `GET /api/kernel/providers` → `{capability: {provider, configured, model}}`.

- [ ] **Step 1:** Write failing tests — `resolve_provider_config("llm", api_key="k")` returns key "k" (param wins); with vault set, vault beats env; with neither, ollama default and no key error; `/api/kernel/providers` shows `configured: true` only for capabilities with vault/env/param present.
- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3:** Move `_resolve_api_base`/`_resolve_api_key`/`resolve_backend_config` into `router.py` generalized per capability; `llm_gateway.py` re-exports the old names (back-compat shim, one line each). Add `kernel.py` route + `include_router` in `main.py`.
- [ ] **Step 4:** Run tests + `uv run pytest tests -q`; PASS.
- [ ] **Step 5:** Commit: `feat(kernel): provider-neutral router with per-capability resolution`

### Task 5: Keys API + Providers UI

**Files:**
- Create: `backend/app/backend/api/keys.py`, `nextjs-frontend/src/app/(main)/settings/providers/page.tsx` (+ client form component; find existing Settings page pattern first)
- Modify: `backend/main.py` (include keys router)
- Test: `tests/backend/api/test_keys_api.py`

**Interfaces:**
- Consumes: `KeyVault` (T3), `router.resolve_provider_config` (T4) for the Test button.
- Produces: `GET /api/keys` (list), `PUT /api/keys {kind, provider, label, value}` (kind validated against `CAPABILITIES`), `DELETE /api/keys/{id}`; `POST /api/kernel/providers/test {capability}` → `{ok, latency_ms?, error?}` via provider `/models`-style endpoint (openrouter/ollama/nvidia), non-implemented capabilities return 501.

- [ ] **Step 1:** Write failing tests — PUT then GET lists the key with `configured: true` and no `value`/`token` field; PUT bad kind → 422; DELETE removes.
- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3:** Implement `keys.py` + test endpoint; build UI form (kind select, provider, label, password input, Save/Delete/Test) on the Settings page.
- [ ] **Step 4:** Run tests; PASS; `cd nextjs-frontend && npm run lint` clean.
- [ ] **Step 5:** Commit: `feat(api,ui): encrypted key management with providers settings page`

### Task 6: uv profiles + lazy heavy imports

**Files:**
- Modify: `pyproject.toml` (replace `[dependency-groups]` backend/voice/all with `core`, `ml`, `voice`, `ocr`, `dev`, `all`), `backend/main.py:35-37` (lazy `RAGService`, `preload_stt_model`, `preload_tts_model` inside lifespan), any `backend/app/services/*.py` / `api/*.py` with top-level chromadb/torch/transformers/pipecat/kokoro/librosa/easyocr/cv2/crewai imports
- Test: `tests/backend/test_profile_imports.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `core` = fastapi, uvicorn, python-multipart, jinja2, pydantic, httpx, requests, python-dotenv, psutil, rank-bm25, asyncpg, cryptography, litellm, ollama; `ml` = chromadb, sentence-transformers, torch, torchvision, transformers, accelerate, huggingface-hub, numpy, pandas, crewai, nltk; `voice` = pipecat-ai, aiortc, pyttsx3, kokoro, spacy model, librosa, soundfile; `ocr` = easyocr, opencv-python-headless, camelot-py[cv], Pillow, PyMuPDF, pdfplumber, python-docx, python-pptx, bs4, trafilatura, striprtf, odfpy, openpyxl, PyYAML; `all` = union.

- [ ] **Step 1:** Write failing test — AST-walk `backend/main.py` + `backend/app/**/*.py`: no top-level (module-body) import of the heavy list; assert `backend/main.py` imports `RAGService`/speech preloads only inside functions.
- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3:** Rewrite pyproject groups; move heavy imports under functions/`TYPE_CHECKING`; keep behavior identical (speech preload still runs in lifespan).
- [ ] **Step 4:** Run test; PASS. Verification: `uv sync --group core && uv run python -c "import backend.main; print('ok')"`.
- [ ] **Step 5:** Commit: `build(uv): split deps into core/ml/voice/ocr profiles, lazy heavy imports`

### Task 7: Docker multi-stage + compose cleanup

**Files:**
- Rewrite: `infra/docker/Dockerfile`
- Modify: `infra/docker/compose/docker-compose.yml`, delete `docker-compose.auth.yml`, `.env.example`, `Makefile` if it references auth overlay

**Interfaces:**
- Consumes: uv profiles (T6).
- Produces: builder stage installs wheels with `UV_LINK_MODE=copy`; runtime stage copies `/app/.venv` + app only, apt = curl, procps, libsndfile1 (+ ffmpeg -dev + espeak only when `VOICE=1` build arg); `PROFILE` build arg defaults `core,ml`.

- [ ] **Step 1:** Record baseline: `docker images | grep litemindui` into commit-adjacent note (or `docker build` current Dockerfile if absent).
- [ ] **Step 2:** Rewrite Dockerfile per above; compose: drop `GOTRUE_API_URL`, `AUTH_MODE`, `GOTRUE_JWT_SECRET`, `depends_on: [gotrue, db]` → keep `db`; delete auth overlay; `.env.example` drop auth vars, add `KEYVAULT_KEY` optional note.
- [ ] **Step 3:** Verify: `docker build --build-arg PROFILE=core -t litemindui:core .` boots; `curl localhost:8000/health` 200; `docker images` size ≤ 1 GB; `make up` default stack still healthy (RAG needs ml).
- [ ] **Step 4:** Commit: `build(docker): multi-stage core/ml profiles, drop auth stack, shrink base`

### Task 8: Frontend auth removal + leftover cleanup + docs + gate

**Files:**
- Delete: `nextjs-frontend/src/app/auth-provider.tsx`, `nextjs-frontend/src/app/(auth)/`
- Modify: `nextjs-frontend/src/app/layout.tsx:6,52-54`, `CLAUDE.md`, `README.md`, `pyproject.toml` (drop `python-jose[cryptography]` from all three dependency lists), `backend/config.py` (drop `GOTRUE_API_URL`/`GOTRUE_JWT_SECRET` fields + comment), `backend/app/backend/conversation_store.py` (GoTrue mirror comment)
- Test: full gates

**Interfaces:**
- Consumes: T1-T7.
- Produces: auth-free frontend; docs matching new layout (`backend/app/kernel/`, uv profiles, `/api/keys`, `/api/kernel/providers`).

- [ ] **Step 1:** Delete auth files, unwrap `<AuthProvider>` in layout; grep `src` for `auth|supabase|gotrue` (case-insensitive) → zero hits outside `(auth)`-free tree.
- [ ] **Step 2:** Remove `python-jose[cryptography]` from `pyproject.toml` (3 lists: optional-dependencies backend, dependency-groups backend, all); drop `GOTRUE_API_URL`/`GOTRUE_JWT_SECRET` from `backend/config.py` and the GoTrue comment in `conversation_store.py`; verify `uv run python -c "import backend.config"` still passes.
- [ ] **Step 3:** Update CLAUDE.md (commands use uv groups core/ml/voice/ocr; directory layout adds `kernel/`; remove auth rows/tables mentions) and README auth sections.
- [ ] **Step 4:** Final gate: `uv run ruff check . && uv run pytest -q && uv run ty check backend/app backend/main.py` and `cd nextjs-frontend && npm run lint && npm run build`.
- [ ] **Step 5:** Commit: `chore: remove auth surface, config, deps, document kernel layout`
