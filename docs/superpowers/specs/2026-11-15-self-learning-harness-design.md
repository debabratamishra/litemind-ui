# LiteMindUI → Self-Learning Harness — Design

Date: 2026-11-15
Branch: `feat/self-learning-harness`

## Intent

Turn LiteMindUI from a chat+RAG app into a self-learning harness whose
center of gravity is an LLM-provider-neutral kernel. Plugins, skills, and
sub-agents are first-class, loadable at runtime. The system accumulates
user facts, then authored skills, then orchestration know-how. Docker
image stops being huge. Authentication is removed entirely (local-first,
single user).

## Goals

1. Provider-neutral kernel: every capability (chat, VLM, STT, TTS,
   embeddings) resolves through one router; no provider hardcoded.
2. API keys (LLM, VLM, speech, embeddings) supplied and rotated from the
   Next.js UI, stored encrypted, hot-reloaded without restart.
3. Plugin runtime: drop a directory + manifest, plugin registers skills,
   tools, and models without touching core.
4. Sub-agents: main loop can spawn scoped agents with tool allowlists.
5. Learning phases:
   - P1 user facts (extends existing `user_memory_service`)
   - P2 self-authored skills (write → test → install)
   - P3 orchestration memory (which agent config solved what)
6. Image size: base stage ships `core` profile only; voice/ml/ocr opt-in.
7. Delete auth: `auth*.py`, `python-jose`, Supabase/GoTrue compose +
   frontend bits.

## Non-goals

- Multi-user, teams, cloud hosting.
- MCP protocol support (a plugin adapter may come later; not now).
- GPU provisioning, k8s.

## Anti-goals (failure even if it works)

- Learning that writes code without running the repo's own test gate.
- Keys on disk unencrypted or in git.
- Image bigger than today after the rework.

## Constraints

- Local-first single user. Postgres already present for memory/conversations.
- uv only; `uv run` wrappers for every command.
- One process per compose service; no new message bus.

## Approaches

- **A. Bolt-on**: add layers to current app, one image. Rejected: one
  process owns everything; a bad plugin can kill chat; image stays huge.
- **B. Kernel + profiles** (chosen): provider-neutral kernel + runtime
  registry + uv install profiles (`core`/`ml`/`voice`/`ocr`).
- **C. Microservices mesh**: message bus, agent-per-container. Rejected:
  wrong scale for one user on one machine.

## Architecture

```
nextjs-frontend
   │  /api/kernel/providers, /api/keys, /api/plugins, /api/agents
   ▼
kernel (backend/app/kernel)
   ├── router.py        — provider-neutral capability routing
   ├── keyvault.py      — Postgres-backed, Fernet, key = SECRET_KEY-derived
   ├── registry.py      — plugin/skill/tool discovery + lifecycle
   └── agents.py        — sub-agent runner (scoped LLM calls, tool allowlist)
plugins/<name>/manifest.toml + *.py   (hot-loaded, sandboxed by allowlist)
services/  (rag, speech, search — become plugins over kernel)
```

### Key store (new)

- Table `user_credentials(id, kind, label, provider, ciphertext, nonce,
  updated_at)`; kind ∈ {llm, vlm, stt, tts, embeddings, search}.
- Fernet key derived from `SECRET_KEY` via HKDF; env fallback when unset.
- `GET /api/keys` returns kind/label/provider + `configured: bool` only,
  never ciphertext. PUT encrypts. Delete wipes row.
- Kernel resolves per-request: explicit param > user key store > env var.
  `llm_gateway.resolve_backend_config()` becomes thin wrapper over kernel
  router (back-compat during migration).
- Hot reload: resolution is per-request already; store read per call, no
  cache to invalidate.

### Plugin runtime

- `plugins/<name>/manifest.toml`: name, version, provides [skills|tools|
  models|capabilities], allows (tool allowlist for sub-agents).
- Registry scans at boot and on `POST /api/plugins/reload`.
- Plugins get kernel handles only: llm/chat, vlm, stt, tts, search, rag,
  memory, spawn_agent. No raw os/subprocess/socket.
- Disable = one flag in manifest or `DELETE /api/plugins/<name>`.

### Sub-agents

- Structured LLM call: system prompt + tool allowlist + max turns.
- Spawned in request scope; result is text + artifacts. No new processes.
- P3 adds `agent_runs` table: task pattern, agent config, outcome,
  reuse count; router prefers winning config.

### Learning

- P1: existing extraction (≤3 ops/turn) + correction-driven updates
  ("no, X" → update conflicting memory) + weekly prune of stale facts.
- P2: on repeated task pattern, kernel drafts plugin manifest+code, runs
  `uv run pytest plugins/<name>/test_*.py` gate, installs only on pass.
  Human approves via UI before activate.
- P3: wins table above.

## Docker / image size

- uv profiles: `core` (fastapi, litellm, httpx, pydantic, asyncpg),
  `ml` (torch, transformers, chromadb, sentence-transformers), `voice`
  (pipecat, aiortc, kokoro, librosa), `ocr` (easyocr, opencv, camelot).
- Base stage: build-essential gone after wheel stage (multi-stage),
  git/psmisc/wget gone, ffmpeg `-dev` only in voice stage, espeak only in
  voice stage.
- Compose service profiles: `voice`, `ml`, `ocr` opt-in.
- Acceptance: `docker images` base ≤ 1 GB; today's baseline measured
  before changes (record in PR).

## Auth removal

- Delete `api/auth*.py`, `auth_deps.py`, `auth_verify.py`, `auth_service.py`
  (check call sites first), `python-jose` dep, supabase/gotrue compose
  services, frontend `auth-provider.tsx` + `/auth` routes, `SECRET_KEY`
  reuse for keyvault.
- Single local user; memory rows keep `user_id` column defaulted.

## API surface (new/changed)

- `GET/PUT/DELETE /api/keys` — encrypted CRUD, no plaintext echo.
- `GET /api/kernel/providers` — what's configured, per capability.
- `GET/POST/DELETE /api/plugins`, `POST /api/plugins/reload`.
- `POST /api/agents` — spawn sub-agent, stream result.

## Phases + acceptance

1. Auth gone, kernel extracted, keyvault live, UI key form, image slimmed.
   Gate: `uv run pytest`, `npm run lint`, base image ≤ 1 GB.
2. Plugin runtime + sub-agents + P1 learning. Gate: 1 bundled plugin
   (web search) loaded via registry, sub-agent resolves a task, memory
   correction test passes.
3. Self-authoring skills + P3 wins table. Gate: authored plugin passes
   test gate before activate; wins table reused across sessions.

## Builder-check decisions (settled)

- **uv profile boundary**: `core` = fastapi, uvicorn, litellm, httpx, pydantic,
  asyncpg, python-dotenv, psutil, python-multipart, jinja2, requests, rank-bm25.
  `ml` = chromadb, sentence-transformers, torch, torchvision, transformers,
  accelerate, huggingface-hub, numpy, pandas, crewai, nltk. `voice` = pipecat,
  aiortc deps, kokoro, pyttsx3, librosa, soundfile, espeak, ffmpeg -dev.
  `ocr` = easyocr, opencv, camelot-py[cv], Pillow, PyMuPDF, pdfplumber,
  docx/pptx/xlsx parsers. Default compose runs core+ml (RAG keeps working,
  chromadb top-level import stays in ml profile). Slim image = core only.
- **Keyvault key lifecycle**: first boot generates random Fernet key, persists
  to `storage/keyvault.key` (0600, gitignored). `SECRET_KEY` no longer the
  source. Rotation documented as wipe-and-re-enter.
- **users table after auth removal**: lifespan seeds constant system user
  `00000000-0000-0000-0000-000000000001`; FK kept, memory keeps working.
- **VLM scope**: kernel capability entry + provider list only in P1; no new
  UI consumer. Vision models reachable via chat once configured.
- **P3 matching**: embedding similarity over task text; agent config =
  {backend, model, tool_allowlist, system_prompt_hash, max_turns}.
- **`/api/kernel/providers` "configured"**: key/endpoint present only; live
  validation behind explicit "Test" button per provider.
- **Plugin sandbox (chosen option 2)**: trust + human approval. Authored
  plugin code runs in-process; kernel registry blocks `os`, `subprocess`,
  `socket`, and `importlib` loads outside the plugin dir at import time.
  Authored plugins must ship `test_*.py`; `uv run pytest plugins/<name>/`
  is the install gate and ships in the prod image. Human approves in UI
  before activate.

## Left to the builder

- Exact manifest schema fields (suggestion above).
- Which services become plugins first (suggest: rag, web_search, speech).
- Frontend key-UI placement (Settings → Providers).
