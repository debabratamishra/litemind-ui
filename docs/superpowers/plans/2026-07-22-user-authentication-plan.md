# Implementation Plan: User Authentication & Per-User Session Isolation

**Spec**: `docs/superpowers/specs/2026-07-22-user-authentication-design.md` (Approved)
**Date**: 2026-07-22
**Worktree**: `hazy-enchanting-squid`

---

## Header

### Goal
Add an email/password login + registration flow to LiteMindUI using self-hosted Supabase Auth (GoTrue). Every user's chat sessions, conversation history, and memory context must be fully isolated. The flow must be smooth (no extra logins once authenticated) and must work in both **standalone** and **Docker** modes, supporting multiple clients (Next.js web, `litemind-cli`, future `litemind-desktop`).

### Architecture
```
Next.js Web (3000)  ─┐
                     ├─ HTTP ─▶ FastAPI Auth Router ─▶ GoTrue (JWT) + Postgres (users/convos)
litemind-cli/desktop ─┘        FastAPI Chat/RAG/Voice ─▶ get_current_user (JWT → user_id) → user-scoped sessions
```
- **Auth backend**: self-hosted GoTrue (standalone binary OR Docker service).
- **Session**: Hybrid — HTTP-only cookie (`access_token`) for browsers + JWT in response body for CLI/desktop.
- **Per-user isolation**: JWT `sub` (GoTrue user id) namespaces in-memory chat sessions (`{user_id}:{session_id}`) and Postgres conversation rows (`WHERE user_id = ...`).
- **RAG vectors**: remain in ChromaDB (unchanged).

### Tech Stack
- Backend: FastAPI, `httpx` (async GoTrue REST calls + test `MockTransport`), `python-jose[cryptography]` (JWT verify), `asyncpg` (Postgres).
- Frontend: Next.js 16 App Router, React 19, TypeScript strict, Zustand, shadcn/ui + Radix, Tailwind v4.
- Ops: Docker Compose (GoTrue + Postgres), Makefile (standalone GoTrue), `.env.example`.

### Global Constraints (do not violate)
- Never put backend secrets in `NEXT_PUBLIC_*` vars.
- Do not commit `.env`; update `.env.example` instead.
- Do not hand-edit `src/components/ui/*` (shadcn auto-generated).
- Do not use Pages Router or `getServerSideProps`.
- Version bumps only via `python3 scripts/version.py bump`.
- Voice is a separate pipeline — never route through the skill layer.
- Run `uv run ruff check .` + `uv run ty check ...` before marking any Python change done.
- Run `npm run lint` in `nextjs-frontend/` before marking any TS change done.

### Deviations from spec (intentional, improvements)
1. **`asyncpg` instead of `psycopg2`** for Postgres. The spec's dependency list already names `asyncpg >= 0.29.0`; it matches the backend's existing async patterns. psycopg2 note in the spec's "Database Migration" section is superseded.
2. **`httpx` + `python-jose` instead of the `gotrue` pip package.** We call GoTrue's REST API directly with `httpx` (easily mockable via `MockTransport` in tests) and verify JWTs with `python-jose`. Simpler, less magic, fully testable. (`python-jose` remains per spec; `gotrue`/`passlib` are dropped.)
3. **Real conversation REST persistence.** `conversation_db.py` SQLite layer is currently imported nowhere (conversations live client-side in Zustand). We add a `ConversationStore` (Postgres, asyncpg) + `/api/conversations` CRUD endpoints so user data actually persists server-side and is isolated — this also fixes the SQLite wipe-on-restart bug permanently.

---

## Tasks

### Task 1 — Backend deps + config/env vars
**Files**: `pyproject.toml`, `.env.example`, `app/backend/core/config.py`

1. Add to `[project.optional-dependencies]` `backend` AND `all` groups:
   ```toml
   "httpx",
   "python-jose[cryptography]>=3.3.0",
   "asyncpg>=0.29.0",
   ```
2. Add to `.env.example`:
   ```env
   # ── Supabase Auth / GoTrue ──
   GOTRUE_API_URL=http://localhost:9999
   GOTRUE_JWT_SECRET=change-me-to-a-long-random-string
   AUTH_MODE=docker            # "docker" or "standalone"

   # ── PostgreSQL (users + conversations) ──
   POSTGRES_PASSWORD=postgres
   POSTGRES_USER=postgres
   POSTGRES_DB=postgres
   DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres

   # ── SMTP (optional, for future email verify/reset) ──
   SMTP_HOST=
   SMTP_PORT=587
   SMTP_USER=
   SMTP_PASS=
   SMTP_ADMIN_EMAIL=admin@litemind.local
   ```
3. In `app/backend/core/config.py`, add to `Config` (read via `os.getenv` / `pydantic Settings` if present):
   ```python
   GOTRUE_API_URL: str = os.getenv("GOTRUE_API_URL", "http://localhost:9999")
   GOTRUE_JWT_SECRET: str = os.getenv("GOTRUE_JWT_SECRET", "")
   AUTH_MODE: str = os.getenv("AUTH_MODE", "standalone")
   DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres")
   ```
4. `uv sync --group all` (or `--group backend`).
5. **Verify**: `uv run ruff check .` passes; config values importable.

### Task 2 — JWT verification helper
**Files**: `app/backend/api/auth_verify.py` (new), `tests/test_auth_verify.py` (new)

**Test (fail first)**:
```python
import jwt as jose_jwt
from app.backend.api import auth_verify

def make_token(sub="user-123", secret="s3cr3t"):
    return jose_jwt.encode({"sub": sub, "email": "a@b.com"}, secret, algorithm="HS256")

def test_verify_valid():
    tok = make_token(secret="s3cr3t")
    data = auth_verify.verify_access_token(tok, secret="s3cr3t")
    assert data["sub"] == "user-123"

def test_verify_invalid_secret_raises():
    tok = make_token(secret="s3cr3t")
    import pytest
    with pytest.raises(Exception):
        auth_verify.verify_access_token(tok, secret="wrong")

def test_verify_malformed_raises():
    import pytest
    with pytest.raises(Exception):
        auth_verify.verify_access_token("not.a.jwt", secret="s3cr3t")
```

**Implement** `app/backend/api/auth_verify.py`:
```python
from jose import JWTError, jwt
from app.backend.core.config import Config

def verify_access_token(token: str, secret: str | None = None) -> dict:
    """Verify a GoTrue HS256 JWT and return its claims. Raises on failure."""
    secret = secret or Config.GOTRUE_JWT_SECRET
    if not secret:
        raise ValueError("GOTRUE_JWT_SECRET is not configured")
    try:
        return jwt.decode(token, secret, algorithms=["HS256"])
    except JWTError as e:
        raise ValueError(f"Invalid token: {e}") from e
```
**Verify**: `uv run pytest tests/test_auth_verify.py -q` passes; `ruff`/`ty` clean.

### Task 3 — GoTrue auth service (httpx)
**Files**: `app/backend/api/auth_service.py` (new), `tests/test_auth_service.py` (new)

**Test (fail first)** — mock GoTrue with `httpx.MockTransport`:
```python
import httpx
from app.backend.api import auth_service

def _mock_handler(request):
    if request.url.path == "/token":
        return httpx.Response(200, json={"access_token": "tok", "token_type": "bearer",
                                         "user": {"id": "u1", "email": "a@b.com"}})
    if request.url.path == "/signup":
        return httpx.Response(200, json={"access_token": "tok", "user": {"id": "u1", "email": "a@b.com"}})
    if request.url.path == "/user":
        return httpx.Response(200, json={"id": "u1", "email": "a@b.com"})
    if request.url.path == "/logout":
        return httpx.Response(200, json={})
    return httpx.Response(404)

def test_login():
    svc = auth_service.GoTrueAuthService(client=httpx.Client(transport=httpx.MockTransport(_mock_handler)))
    res = svc.login("a@b.com", "pw")
    assert res["access_token"] == "tok" and res["user"]["id"] == "u1"

def test_register():
    svc = auth_service.GoTrueAuthService(client=httpx.Client(transport=httpx.MockTransport(_mock_handler)))
    res = svc.register("a@b.com", "pw")
    assert res["user"]["id"] == "u1"

def test_get_user():
    svc = auth_service.GoTrueAuthService(client=httpx.Client(transport=httpx.MockTransport(_mock_handler)))
    assert svc.get_user("tok")["id"] == "u1"

def test_logout():
    svc = auth_service.GoTrueAuthService(client=httpx.Client(transport=httpx.MockTransport(_mock_handler)))
    assert svc.logout("tok") is True
```

**Implement** `auth_service.py` (`GoTrueAuthService` with `login`, `register`, `logout`, `get_user`; `base_url` from `Config.GOTRUE_API_URL`; injectable `httpx.Client` for tests). `login` hits `POST /token?grant_type=password`; `register` hits `POST /signup`; `get_user` hits `GET /user` with `Bearer`; `logout` hits `POST /logout` with `Bearer`. Validate `status_code`, raise `auth_service.GoTrueError` (message from GoTrue `msg`/`error_description`) on non-2xx.
**Verify**: tests pass; lint clean.

### Task 4 — Auth dependencies (User model + get_current_user)
**Files**: `app/backend/api/auth_deps.py` (new), `tests/test_auth_deps.py` (new)

**Test (fail first)**:
```python
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi import Depends, HTTPException
from app.backend.api import auth_deps, auth_verify

def _app():
    a = FastAPI()
    @a.get("/me")
    def me(u=Depends(auth_deps.get_current_user)):
        return {"id": u.id, "email": u.email}
    return a

def test_missing_token():
    c = TestClient(_app())
    assert c.get("/me").status_code == 401

def test_bearer_token():
    import jwt as j
    token = j.encode({"sub": "u9", "email": "x@y.com"}, "s3cr3t", algorithm="HS256")
    c = TestClient(_app())
    # patch secret
    auth_deps.Config.GOTRUE_JWT_SECRET = "s3cr3t"
    r = c.get("/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200 and r.json()["id"] == "u9"

def test_cookie_token():
    import jwt as j
    token = j.encode({"sub": "u9", "email": "x@y.com"}, "s3cr3t", algorithm="HS256")
    auth_deps.Config.GOTRUE_JWT_SECRET = "s3cr3t"
    c = TestClient(_app())
    r = c.get("/me", cookies={"access_token": token})
    assert r.status_code == 200 and r.json()["id"] == "u9"
```

**Implement** `auth_deps.py`:
```python
from typing import Optional
from fastapi import Depends, HTTPException, status, Cookie
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from app.backend.api.auth_verify import verify_access_token
from app.backend.core import config as Config

security = HTTPBearer(auto_error=False)

class User(BaseModel):
    id: str
    email: Optional[str] = None

def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    access_token: Optional[str] = Cookie(None),
) -> User:
    token = credentials.credentials if credentials else access_token
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    try:
        claims = verify_access_token(token)
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    sub = claims.get("sub")
    if not sub:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token missing subject")
    return User(id=sub, email=claims.get("email"))
```
**Verify**: tests pass; lint clean.

### Task 5 — Auth router
**Files**: `app/backend/api/auth.py` (new), `tests/test_auth_router.py` (new)

**Test (fail first)** using `app.dependency_overrides` + mocked `GoTrueAuthService`:
```python
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.backend.api import auth

class FakeAuth:
    def login(self, e, p): return {"access_token": "tok", "user": {"id": "u1", "email": e}}
    def register(self, e, p): return {"access_token": "tok", "user": {"id": "u1", "email": e}}
    def logout(self, t): return True
    def get_user(self, t): return {"id": "u1", "email": "a@b.com"}

def _client():
    a = FastAPI(); a.include_router(auth.router)
    auth.get_auth_service = lambda: FakeAuth()   # override factory
    return TestClient(a)

def test_register():
    r = _client().post("/api/auth/register", json={"email": "a@b.com", "password": "pw"})
    assert r.status_code == 200 and r.json()["access_token"] == "tok"

def test_login_sets_cookie_and_body():
    r = _client().post("/api/auth/login", json={"email": "a@b.com", "password": "pw"})
    assert r.status_code == 200
    assert r.cookies.get("access_token") == "tok"
    assert r.json()["access_token"] == "tok"

def test_me_requires_auth():
    r = _client().get("/api/auth/me")
    assert r.status_code == 401

def test_me_with_cookie():
    c = _client()
    # set cookie by logging in first
    c.post("/api/auth/login", json={"email": "a@b.com", "password": "pw"})
    r = c.get("/api/auth/me")
    assert r.status_code == 200 and r.json()["email"] == "a@b.com"

def test_logout_clears_cookie():
    c = _client()
    c.post("/api/auth/login", json={"email": "a@b.com", "password": "pw"})
    r = c.post("/api/auth/logout")
    assert r.status_code == 200
    assert "access_token" in r.cookies and r.cookies["access_token"] == ""
```
**Implement** `auth.py`: `APIRouter(prefix="/api/auth")`. `get_auth_service()` factory (default `GoTrueAuthService()`) so tests override it. Endpoints:
- `POST /register` → service.register → return `{access_token, token_type, user}`, set cookie if successful.
- `POST /login` → service.login → set `Response` cookie `access_token` (httponly, samesite=lax, max_age extended if `remember`) AND return `{access_token, user}`.
- `POST /logout` → service.logout(token) → clear cookie.
- `GET /me` → `Depends(get_current_user)` → return user dict.
Map service errors to 401/409/503 (see Error Handling in spec).
**Verify**: tests pass; lint clean. Register router in `main.py` (`app.include_router(auth.router)`).

### Task 6 — Fix conversation_db.py wipe bug
**Files**: `app/services/conversation_db.py`

Remove the destructive calls:
- Delete `clear_database_file()` invocation at module import (line ~36).
- Delete `atexit.register(clear_database_file)` (line ~39).
Keep `clear_database_file` function defined but **never auto-invoked** (it may still be used by an explicit reset endpoint/test).
**Verify**: `uv run ty check app/services` clean. Add a quick `tests/test_conversation_db_no_wipe.py` asserting the module imports without deleting the file:
```python
from pathlib import Path
import app.services.conversation_db as cdb
def test_import_does_not_delete_db(tmp_path, monkeypatch):
    p = tmp_path / "conversations.db"; p.write_text("data")
    monkeypatch.setattr(cdb, "DEFAULT_DB_PATH", p)
    # re-import fresh
    import importlib; importlib.reload(cdb)
    assert p.exists() and p.read_text() == "data"
```

### Task 7 — Postgres conversation store
**Files**: `app/backend/conversation_store.py` (new), `tests/test_conversation_store.py` (new)

Schema (run via an `init_schema()` coroutine that executes DDL; idempotent `CREATE TABLE IF NOT EXISTS`):
```sql
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL DEFAULT 'New Chat',
    conversation_type TEXT DEFAULT 'chat',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    summary TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
```
**Test (fail first)** — use `asyncpg` against a test Postgres OR mock the pool. Simplest reliable approach: factor DB access behind a `ConversationStore` whose `self.pool` is injectable; test with an in-memory fake implementing the same coroutine methods, OR run against a real test DB if available. Provide a `FakePool` test double to keep `pytest` hermetic:
```python
import pytest
from app.backend import conversation_store as cs

class FakeStore(cs.ConversationStore):
    """In-memory test double: dict-backed, implements the same async method
    signatures as the real asyncpg-backed store, filtering every query by user_id."""
    def __init__(self): self.rows = {}  # conversation_id -> (user_id, title, messages[])
    async def create_conversation(self, user_id, title): ...  # dict-backed
    # ... in-memory versions of each store method (get/list/update/delete/messages)

@pytest.mark.asyncio
async def test_create_and_list_isolated():
    s = FakeStore()
    c1 = await s.create_conversation("u1", "A")
    c2 = await s.create_conversation("u2", "B")
    assert len(await s.list_conversations("u1")) == 1
    assert (await s.list_conversations("u1"))[0].id == c1.id
    assert len(await s.list_conversations("u2")) == 1
```
**Implement** `ConversationStore` with `asyncpg.create_pool(DSN)`, `get_conversation_store()` singleton, and methods: `init_schema`, `upsert_user`, `create_conversation(user_id, title)`, `get_conversation(conversation_id, user_id)`, `list_conversations(user_id)`, `update_conversation(...)`, `delete_conversation(conversation_id, user_id)`, `add_message(conversation_id, user_id, role, content)`, `get_messages(conversation_id, user_id)`, `delete_message(...)`. **Every query filters by `user_id`** (isolation). `GOTRUE_JWT_SECRET`/`DATABASE_URL` from `Config`.
**Verify**: tests pass; `ty`/`ruff` clean.

### Task 8 — Conversation REST endpoints
**Files**: `app/backend/api/conversations.py` (new), `tests/test_conversations_api.py` (new)

All routes protected via `Depends(get_current_user)`. Endpoints:
- `GET /api/conversations` → list for `user.id`
- `POST /api/conversations` → create `{title}` → returns conversation
- `GET /api/conversations/{id}` → get (404 if not owner)
- `PATCH /api/conversations/{id}` → rename/summary
- `DELETE /api/conversations/{id}` → delete (404 if not owner)
- `GET /api/conversations/{id}/messages` → messages
- `POST /api/conversations/{id}/messages` → append `{role, content}`

**Test (fail first)**: use `TestClient` with `dependency_overrides[get_current_user]` returning `User(id="u1")`, plus a fake `ConversationStore`:
```python
def test_list_empty(): ...
def test_create_and_get_isolated():
    # u1 creates; u2 override cannot fetch u1's conversation (404)
```
**Verify**: tests pass; register router in `main.py`.

### Task 9 — Protect chat endpoints + memory isolation
**Files**: `app/backend/api/chat.py`, `app/services/conversation_memory.py`

- In `conversation_memory.py`: change `get_or_create_session(self, session_id)` → `get_or_create_session(self, user_id: str, session_id: str)`, building `scoped = f"{user_id}:{session_id}"`. Update `clear_session`, `get_session_stats`, `prepare_context_for_llm`, `summarize_if_needed` signatures to accept `user_id`. Keep `get_memory_service()` singleton.
- In `chat.py`: inject `user: User = Depends(get_current_user)` on `/api/chat`, `/api/chat/stream`, `/api/chat/web-search`. Pass `user.id` into `_build_messages_with_history`/memory calls using scoped session id `f"{user.id}:{session_id}"`. Update the three memory endpoints (`/api/chat/memory/stats/{session_id}` etc.) to also require auth and scope with `user.id`.

**Test (fail first)** `tests/test_chat_auth.py`:
```python
def test_stream_requires_auth():
    # TestClient(main:app) POST /api/chat/stream without auth → 401
def test_memory_isolation_between_users():
    # uA and uB with same session_id → distinct sessions, no leakage
```
**Verify**: tests pass; `ty`/`ruff` clean.

### Task 10 — Protect RAG + voice endpoints
**Files**: `main.py` (`/api/rag/query`), `app/backend/api/voice.py` (`/api/voice/offer`)

- `/api/rag/query`: inject `user: User = Depends(get_current_user)`. (Optionally namespace RAG session/summary with `user.id` the same way as chat.)
- `/api/voice/offer`: inject `Depends(get_current_user)`; pass `user.id` into pipeline context for transcript isolation.

**Test (fail first)** `tests/test_rag_voice_auth.py`:
```python
def test_rag_query_requires_auth(): ...
def test_voice_offer_requires_auth(): ...
```
**Verify**: tests pass; lint clean.

### Task 11 — Docker Compose + Makefile + env
**Files**: `docker-compose.yml`, `Makefile` (or `Makefile.auth`), `.env.example` (Task 1), `README`/docs

- Add `gotrue` + `db` (postgres:15-alpine) services to `docker-compose.yml` with env from spec §Self-Hosting. Wire `backend` `depends_on: [gotrue, db]` and `GOTRUE_API_URL=http://gotrue:9999`, `DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@db:5432/postgres`, `AUTH_MODE=docker`.
- Add Makefile targets: `make gotrue-up` (standalone: pull+run `supabase/gotrue` binary with env from `.env`), `make gotrue-down`. Document both modes in `docs/superpowers/specs/...` (already in spec) + a short README note.
- Expose GoTrue port `9999` and Postgres `5432` for local dev.

**Verify**: `docker compose config` validates; `make gotrue-up` (if Docker available) starts GoTrue; lint unaffected.

### Task 12 — Frontend auth store + API client
**Files**: `nextjs-frontend/src/lib/store.ts`, `nextjs-frontend/src/lib/api.ts`

- `store.ts`: add `AuthState` slice (typed `User`, `accessToken`, `isAuthenticated`, `isLoading`, `login`, `register`, `logout`, `fetchCurrentUser`). Token kept in memory only (not localStorage).
- `api.ts`: add `credentials: 'include'` to every `fetch` (cookie auth for web). Add helper `authFetch` that also injects `Authorization: Bearer <token>` when `accessToken` present (CLI/desktop parity). Add `authApi = { login, register, logout, me }`.

**Test (fail first)** — type-check + a Zustand unit test (`auth.test.ts`): login sets `isAuthenticated`, logout clears.
**Verify**: `npm run lint` + `npx tsc --noEmit` clean.

### Task 13 — Route groups + login/register pages
**Files**:
- `nextjs-frontend/src/app/(auth)/login/page.tsx`
- `nextjs-frontend/src/app/(auth)/register/page.tsx`
- `nextjs-frontend/src/app/(main)/layout.tsx` (wraps existing chrome: ThemeProvider/Tooltip/Sidebar/main)
- `nextjs-frontend/src/app/(main)/page.tsx` (existing home moved here) + move `/chat`, `/rag` under `(main)`.
- Keep `layout.tsx` (root) minimal; `(main)` adds the sidebar chrome.

Login/Register pages: shadcn `Card`, `Input`, `Button`, `Label`; email/password fields; "Remember me"; links between; client-side validation; call `authApi.login/register` then `router.push('/chat')`. Test in **light + dark** mode.
**Verify**: `npm run lint`, `npm run build` succeed.

### Task 14 — AuthProvider guard + sidebar
**Files**: `nextjs-frontend/src/app/auth-provider.tsx` (client, wraps providers, calls `fetchCurrentUser()` on mount), `nextjs-frontend/src/components/layout/sidebar.tsx` (auth section), `nextjs-frontend/src/components/auth/protected-route.tsx`

- `auth-provider.tsx`: on mount, `fetchCurrentUser()` to rehydrate session; renders children.
- `protected-route.tsx`: if `!isAuthenticated && !isLoading` → `router.replace('/login')`; while loading show spinner.
- Sidebar: authenticated → avatar/email + "Sign out"; else → "Sign in" button → `/login`.
**Verify**: `npm run lint`, `npm run build` succeed.

### Task 15 — Cross-cutting: lint, build, tests, docs
- `uv run ruff check .` and `uv run ty check app/backend app/services app/core` clean.
- `uv run pytest tests/ -q` green (auth, auth_deps, auth_router, conversation isolation, chat auth, rag/voice auth, conversation_store, conversations_api).
- `cd nextjs-frontend && npm run lint && npm run build` green.
- Update `README.md` with auth setup (standalone + docker) + env vars.
- Add a short `docs/superpowers/specs/...` note linking plan↔spec (already in spec §Self-Hosting).
- Commit per task; final PR labelled `minor` (new feature) → `python3 scripts/version.py bump minor` only when user requests release.

---

## Commit Cadence
Commit after each task with a clear message, e.g.:
- `feat(auth): add JWT verify + GoTrue service + auth deps`
- `feat(auth): auth router register/login/logout/me`
- `fix: remove SQLite wipe-on-restart in conversation_db`
- `feat(auth): Postgres conversation store + REST CRUD`
- `feat(auth): protect chat/rag/voice endpoints with get_current_user`
- `feat(auth): docker compose gotrue+postgres + make targets`
- `feat(auth): frontend store, pages, guard, sidebar`

## Verification Gate (all must pass before "done")
1. `uv run ruff check .` ✅
2. `uv run ty check app/backend app/services app/core` ✅
3. `uv run pytest tests/ -q` ✅
4. `cd nextjs-frontend && npm run lint && npm run build` ✅
5. Manual: register → login (cookie set) → chat → refresh page (session rehydrates) → second user cannot see first user's conversations.
