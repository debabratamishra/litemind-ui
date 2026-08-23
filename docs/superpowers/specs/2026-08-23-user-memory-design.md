# Persistent User Memory — Design

**Date:** 2026-08-23
**Status:** Approved (approach A + Postgres + doc updates)
**Scope:** Personalization via durable, user-level memory that persists across sessions and is injected into fresh conversations on all surfaces and all LLM providers.

## Problem

LiteMindUI remembers context only within a session (`conversation_memory.py`, session-scoped). When a user starts a fresh conversation, everything they previously taught the assistant (preferences, identity, ongoing projects) is gone. We need persistent per-user memory: captured automatically from conversations plus explicitly manageable by the user.

## Goals

1. Memories survive sessions and are available at the start of any new conversation.
2. Zero-effort capture: durable facts stated in normal conversation get saved automatically.
3. Explicit control: "remember that …" works mid-conversation; users can view/add/edit/delete memories in Settings.
4. Provider-agnostic: identical behavior on Ollama, OpenRouter, and Nvidia NIM.
5. Surfaces: chat, RAG, and realtime voice all read (and voice/chat write) memory.

## Non-goals (v1)

- No vector/semantic retrieval of memories (inject-all-capped is sufficient below ~100 memories/user; upgrade path later).
- No global on/off toggle in UI.
- No cross-user or shared memory; no memory for unauthenticated use (all routes already require auth).
- No migration of legacy SQLite `conversation_db.py`.

## Naming

Existing `conversation_memory.py` and `/api/chat/memory/*` are **session-scoped** and stay untouched. The new feature is consistently named **user memory**: `user_memory_store.py`, `user_memory_service.py`, `/api/memory/*`, table `user_memories`.

## Architecture

Three units, one dependency direction:

```
app/backend/user_memory_store.py    # persistence (asyncpg)  ← app/services/user_memory_service.py
app/services/user_memory_service.py # extraction + formatting (LLM ops via llm_gateway)
app/backend/api/memory.py           # CRUD router (auth-required)
```

Surfaces consume the service; none talk to the store directly.

### 1. Storage — `app/backend/user_memory_store.py`

Mirrors `conversation_store.py`: lazy `asyncpg.create_pool` singleton, `init_schema()` idempotent DDL, module-level `get_user_memory_store()`. DSN from `Config.DATABASE_URL`.

```sql
CREATE TABLE IF NOT EXISTS user_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'auto',   -- 'auto' | 'manual'
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_user_memories_user_id ON user_memories(user_id);
```

API of the store class (every method user-scoped — same isolation guarantee as conversations):

| Method | Returns |
|---|---|
| `list_memories(user_id, limit=50)` | newest-first `List[UserMemoryRecord]` |
| `add_memory(user_id, content, source="auto")` | `UserMemoryRecord` |
| `update_memory(user_id, memory_id, content)` | `bool` |
| `delete_memory(user_id, memory_id)` | `bool` |
| `clear_memories(user_id)` | count deleted |

`ON DELETE CASCADE` from `users` means GoTrue account deletion wipes memories automatically. `UserMemoryRecord` dataclass with `to_dict()` follows `ConversationRecord`.

### 2. Memory service — `app/services/user_memory_service.py`

Pure functions + one orchestrator; no FastAPI imports.

- `build_memory_block(memories: List[...]) -> str` — returns a system-prompt fragment:

  ```
  About the user (persistent memory; use when relevant):
  - <content>
  - <content>
  ```

  Empty string when the list is empty. Caller embeds it as a `system` message.

- `extract_memory_ops(user_message, assistant_message, existing_memories, llm_config) -> List[dict]`
  Calls `llm_gateway.complete_text()` with an extraction prompt containing:
  - the latest user message and assistant reply,
  - the existing memory list (id + content),
  - rules: return strict JSON array only; each op is one of
    `{"op":"add","content":str}` / `{"op":"update","id":str,"content":str}` / `{"op":"delete","id":str}`;
  - what qualifies: durable facts only — identity, stable preferences, ongoing projects/goals, corrections of stored facts, explicit "remember that…" requests;
  - what never qualifies: transient task talk, secrets/credentials/API keys/passwords, opinions about the assistant;
  - ≤3 ops per turn; empty array `[]` when nothing durable;
  - prefer `update` of an existing memory over adding a near-duplicate.

  Parsing: strip code fences if present, `json.loads`, validate shape, drop invalid ops silently. Cap enforced in code (first 3 valid ops).

- `apply_memory_ops(store, user_id, ops) -> None` — maps ops to store calls; unknown ids ignored.

- `run_memory_update(user_id, exchange..., llm_config) -> None` — orchestrator meant to run as a fire-and-forget task: extract → apply. All exceptions logged at WARNING, never propagated.

### 3. API — `app/backend/api/memory.py`

Router mounted in `main.py`. All endpoints `Depends(get_current_user)`; Pydantic models added to `api_models.py`.

| Endpoint | Purpose |
|---|---|
| `GET /api/memory` | list memories |
| `POST /api/memory` | add `{content}` → saved with `source="manual"` |
| `PUT /api/memory/{id}` | edit content |
| `DELETE /api/memory/{id}` | delete one |
| `POST /api/memory/clear` | delete all for user |

404 on foreign/unknown ids (user scoping makes ownership implicit).

### 4. Wiring

**Read path (injection):**

- Chat: `_build_messages_with_history()` in `chat.py` loads memories via store and prepends `build_memory_block` as a `system` message before other system messages.
- RAG: the RAG answer-composition prompt gains the same block (one call site in `rag_service.py` / rag skill).
- Voice: the pipeline's LLM service system prompt gains the block (loaded once at pipeline start; a long-lived session picks up changes on reconnect — accepted limitation).
- The client sends nothing new; the backend resolves memories from the JWT `user.id`.

**Write path (extraction trigger):**

- `chat_endpoint`: after a successful response, schedule `asyncio.create_task(run_memory_update(...))`.
- `chat_stream`: the event generator accumulates streamed assistant text; after the stream completes normally, schedules the same task. Aborted/errored streams skip extraction.
- Voice: after each completed LLM turn in `voice_pipeline.py`, same hook using the turn's transcript pair.
- Extraction uses the **request's own backend/model/api config** through the gateway — no separate extraction-model setting.

### 5. Error handling

| Failure | Behavior |
|---|---|
| Extraction LLM call fails / times out (~15 s) | log WARNING, skip; chat response unaffected |
| Unparseable extraction JSON | skip silently (log DEBUG) |
| Store write/read fails | log WARNING; read failure degrades to no-memory-block response |
| Fire-and-forget task crash | contained by `run_memory_update`'s catch-all |

Extraction never blocks, delays, or fails the user-visible response.

### 6. Frontend

Settings panel gains a "Personalization / Memory" section: list memories (newest first), inline edit, delete per row, add box, clear-all with confirmation. Uses a thin `memoryApi` addition in `lib/api.ts`. No toggle, no categories v1.

### 7. Documentation updates

Update the project docs so the mechanics stay accurate:

- `CLAUDE.md` (root): directory layout entries for the two new modules + router; architecture overview section describing user memory (capture + injection); fix the stale claim that conversations persist in SQLite — canonical store is Postgres (`conversation_store.py`); document `/api/memory` surface.
- `AGENTS.md` (root): note the user-scoping rule for any new persistence (memories included).
- `README.md`: feature bullet under personalization/memory.
- `nextjs-frontend/CLAUDE.md` and `nextjs-frontend/AGENTS.md`: settings memory panel + `memoryApi` client notes.

### 8. Testing

Backend (`tests/test_user_memory*.py`):

- Ops parsing: valid JSON, fenced JSON, malformed JSON → `[]`, op-shape validation, >3 ops capped, secret-like content rejected by prompt contract is *not* re-checked in code (prompt-level rule; documented).
- `build_memory_block`: empty list → `""`; ordering; header text.
- Store CRUD against a test Postgres (or `asyncpg` test DSN), including cross-user isolation (user B cannot read/update/delete user A's memory).
- Injection: `_build_messages_with_history` includes/excludes the memory system message correctly.
- One integration-style happy path with mocked gateway: exchange → expected ops applied.

Frontend: component test for the settings memory panel following existing test patterns.

## Data flow summary

```
Fresh conversation ─► route auth (JWT) ─► store.list_memories(user.id)
                      ─► build_memory_block ─► system message ─► llm_gateway ─► provider

Exchange completes ─► create_task(run_memory_update)
                      ─► complete_text(extraction prompt w/ existing memories)
                      ─► parse ops ─► store.add/update/delete ─► done (logged)
```
