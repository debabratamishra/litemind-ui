# Persistent User Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persistent per-user memory (auto-extracted + manually managed) injected into chat, RAG, and voice conversations on every LLM provider.

**Architecture:** Postgres table `user_memories` (asyncpg, mirroring `conversation_store.py`), a service module that formats memories into a system-prompt block and extracts new memories via `llm_gateway.complete_text` after each exchange, a CRUD router, and thin wiring at each surface. Extraction runs fire-and-forget and never blocks responses.

**Tech Stack:** FastAPI, asyncpg/Postgres, LiteLLM gateway (existing), Next.js 16 + React 19 + vitest/testing-library.

**Spec:** `docs/superpowers/specs/2026-08-23-user-memory-design.md`

## Global Constraints

- Naming is always **user_memory** / `user_memories` — never reuse the session-scoped "memory" names (`conversation_memory`, `/api/chat/memory/*`).
- Every store query filters by `user_id` (same isolation guarantee as `conversation_store.py`).
- All LLM calls (extraction) go through `app/services/llm_gateway.py` with the request's own backend/model config — no provider-specific code anywhere.
- Extraction failures are logged (WARNING) and skipped; they must never delay or fail a chat/RAG/voice response.
- ≤3 memory ops per extraction; memories injected capped at 50 most recent.
- Python: run `uv run ruff check .` and `uv run ty check app/backend app/services main.py` before finishing each backend task. Frontend: `npm run lint` inside `nextjs-frontend/`.
- Do not touch `version.json`.

## File Structure

```
app/backend/user_memory_store.py      CREATE  Postgres store (record + CRUD + singleton)
app/services/user_memory_service.py   CREATE  block formatting, op parsing, extraction, orchestration
app/backend/api/memory.py             CREATE  /api/memory CRUD router
app/backend/models/api_models.py      MODIFY  request/response models for /api/memory
main.py                               MODIFY  include memory router; RAG query loads memory block
app/backend/api/chat.py               MODIFY  inject block; extraction triggers on /api/chat + /api/chat/stream
app/skills/rag.py                     MODIFY  thread memory_block through both RAG skills
app/services/rag_service.py           MODIFY  query(..., memory_block=None) injection
app/services/voice_pipeline.py        MODIFY  inject at pipeline start; extraction per LLM turn
nextjs-frontend/src/lib/types.ts      MODIFY  MemoryRecord type
nextjs-frontend/src/lib/api.ts        MODIFY  memory API client functions
nextjs-frontend/src/components/settings-panel.tsx  MODIFY  Memory section
tests/test_user_memory_store.py       CREATE
tests/test_user_memory_service.py     CREATE
tests/test_memory_api.py              CREATE
tests/test_user_memory_wiring.py      CREATE
nextjs-frontend/src/components/memory-settings.test.tsx  CREATE
```

---

### Task 1: UserMemoryStore (Postgres persistence)

**Files:**
- Create: `app/backend/user_memory_store.py`
- Test: `tests/test_user_memory_store.py`

**Interfaces:**
- Consumes: `Config.DATABASE_URL` from `app.backend.core.config` (same import as `conversation_store.py`).
- Produces: `UserMemoryRecord(id: str, user_id: str, content: str, source: str, created_at: str, updated_at: str)` with `.to_dict() -> dict`; `UserMemoryStore` with async methods `init_schema() -> None`, `list_memories(user_id: str, limit: int = 50) -> List[UserMemoryRecord]`, `add_memory(user_id: str, content: str, source: str = "auto") -> UserMemoryRecord`, `update_memory(user_id: str, memory_id: str, content: str) -> bool`, `delete_memory(user_id: str, memory_id: str) -> bool`, `clear_memories(user_id: str) -> int`; module function `get_user_memory_store() -> UserMemoryStore` (process-wide singleton).

- [ ] **Step 1: Write the failing tests**

```python
"""Unit tests for UserMemoryStore.

Uses an in-memory FakeStore subclass (same approach as
tests/test_conversation_store.py) so no live PostgreSQL is required. The focus
is user isolation: one user can never read or mutate another user's memories.
"""

import uuid

import pytest

from app.backend.user_memory_store import UserMemoryRecord, UserMemoryStore


class FakeUserMemoryStore(UserMemoryStore):
    """In-memory implementation of the same async interface."""

    def __init__(self):
        self.memories = {}  # id -> dict(row fields)

    async def init_schema(self):
        return None

    async def list_memories(self, user_id, limit=50):
        rows = [m for m in self.memories.values() if m["user_id"] == user_id]
        rows.sort(key=lambda m: m["created_at"], reverse=True)
        return [self._record(m) for m in rows[:limit]]

    async def add_memory(self, user_id, content, source="auto"):
        mid = str(uuid.uuid4())
        row = {
            "id": mid, "user_id": user_id, "content": content, "source": source,
            "created_at": "2026-08-23T00:00:00", "updated_at": "2026-08-23T00:00:00",
        }
        self.memories[mid] = row
        return self._record(row)

    async def update_memory(self, user_id, memory_id, content):
        m = self.memories.get(memory_id)
        if not m or m["user_id"] != user_id:
            return False
        m["content"] = content
        m["updated_at"] = "2026-08-23T01:00:00"
        return True

    async def delete_memory(self, user_id, memory_id):
        m = self.memories.get(memory_id)
        if not m or m["user_id"] != user_id:
            return False
        del self.memories[memory_id]
        return True

    async def clear_memories(self, user_id):
        ids = [i for i, m in self.memories.items() if m["user_id"] == user_id]
        for i in ids:
            del self.memories[i]
        return len(ids)

    @staticmethod
    def _record(m):
        return UserMemoryRecord(**m)


USER_A = str(uuid.uuid4())
USER_B = str(uuid.uuid4())


@pytest.mark.asyncio
async def test_add_and_list_scoped_to_user():
    store = FakeUserMemoryStore()
    await store.add_memory(USER_A, "Prefers concise answers")
    await store.add_memory(USER_B, "Works on robotics")
    assert [m.content for m in await store.list_memories(USER_A)] == ["Prefers concise answers"]


@pytest.mark.asyncio
async def test_update_requires_owner():
    store = FakeUserMemoryStore()
    mem = await store.add_memory(USER_A, "Old fact")
    assert await store.update_memory(USER_B, mem.id, "hijack") is False
    assert await store.update_memory(USER_A, mem.id, "New fact") is True
    assert (await store.list_memories(USER_A))[0].content == "New fact"


@pytest.mark.asyncio
async def test_delete_requires_owner():
    store = FakeUserMemoryStore()
    mem = await store.add_memory(USER_A, "fact")
    assert await store.delete_memory(USER_B, mem.id) is False
    assert await store.delete_memory(USER_A, mem.id) is True
    assert await store.list_memories(USER_A) == []


@pytest.mark.asyncio
async def test_clear_only_own():
    store = FakeUserMemoryStore()
    await store.add_memory(USER_A, "a1")
    await store.add_memory(USER_A, "a2")
    await store.add_memory(USER_B, "b1")
    assert await store.clear_memories(USER_A) == 2
    assert len(await store.list_memories(USER_B)) == 1


@pytest.mark.asyncio
async def test_list_respects_limit_and_newest_first():
    store = FakeUserMemoryStore()
    for i in range(5):
        await store.add_memory(USER_A, f"fact {i}")
    rows = await store.list_memories(USER_A, limit=3)
    assert len(rows) == 3
    assert rows[0].content == "fact 4"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_user_memory_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.backend.user_memory_store'`

- [ ] **Step 3: Write the store**

```python
"""
User-scoped persistent memory store backed by PostgreSQL (asyncpg).

Mirrors ``conversation_store.py``: lazy connection-pool singleton, idempotent
schema init, and every query filtered by ``user_id`` so one user can never
read or mutate another user's memories.

Rows are deleted automatically when the owning user is removed from the
``users`` table (ON DELETE CASCADE).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import List, Optional

import asyncpg

from app.backend.core.config import Config

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS user_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'auto',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_memories_user_id
    ON user_memories(user_id);
"""


@dataclass
class UserMemoryRecord:
    id: str
    user_id: str
    content: str
    source: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "content": self.content,
            "source": self.source,
            "created_at": str(self.created_at),
            "updated_at": str(self.updated_at),
        }


class UserMemoryStore:
    """CRUD access to the ``user_memories`` table, always scoped by user."""

    def __init__(self, dsn: Optional[str] = None):
        self._dsn = dsn or Config.DATABASE_URL
        self._pool: Optional[asyncpg.Pool] = None

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=10)
        return self._pool

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def init_schema(self) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(SCHEMA_SQL)

    @staticmethod
    def _record(row: asyncpg.Record) -> UserMemoryRecord:
        return UserMemoryRecord(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            content=row["content"],
            source=row["source"],
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    async def list_memories(self, user_id: str, limit: int = 50) -> List[UserMemoryRecord]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, user_id, content, source, created_at, updated_at
                FROM user_memories WHERE user_id = $1
                ORDER BY created_at DESC LIMIT $2
                """,
                uuid.UUID(user_id),
                limit,
            )
        return [self._record(r) for r in rows]

    async def add_memory(self, user_id: str, content: str, source: str = "auto") -> UserMemoryRecord:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO user_memories (user_id, content, source)
                VALUES ($1, $2, $3)
                RETURNING id, user_id, content, source, created_at, updated_at
                """,
                uuid.UUID(user_id),
                content,
                source,
            )
        return self._record(row)

    async def update_memory(self, user_id: str, memory_id: str, content: str) -> bool:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE user_memories SET content = $3, updated_at = NOW()
                WHERE id = $2 AND user_id = $1
                RETURNING id
                """,
                uuid.UUID(user_id),
                uuid.UUID(memory_id),
                content,
            )
        return row is not None

    async def delete_memory(self, user_id: str, memory_id: str) -> bool:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "DELETE FROM user_memories WHERE id = $2 AND user_id = $1 RETURNING id",
                uuid.UUID(user_id),
                uuid.UUID(memory_id),
            )
        return row is not None

    async def clear_memories(self, user_id: str) -> int:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM user_memories WHERE user_id = $1", uuid.UUID(user_id)
            )
        # asyncpg returns e.g. "DELETE 5"
        try:
            return int(result.split()[-1])
        except (ValueError, IndexError):
            return 0


_store_instance: Optional[UserMemoryStore] = None


def get_user_memory_store() -> UserMemoryStore:
    """Get the process-wide UserMemoryStore singleton."""
    global _store_instance
    if _store_instance is None:
        _store_instance = UserMemoryStore()
    return _store_instance
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_user_memory_store.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check app/backend/user_memory_store.py tests/test_user_memory_store.py
uv run ty check app/backend
git add app/backend/user_memory_store.py tests/test_user_memory_store.py
git commit -m "feat: add user-scoped memory store (Postgres)"
```

---

### Task 2: Memory service — block formatting + op parsing

**Files:**
- Create: `app/services/user_memory_service.py`
- Test: `tests/test_user_memory_service.py`

**Interfaces:**
- Consumes: `UserMemoryRecord` / `UserMemoryStore` from Task 1.
- Produces:
  - `build_memory_block(memories: List[Any]) -> str` — accepts objects with `.content`; returns `""` when list is empty, else the block below.
  - `parse_memory_ops(raw: str) -> List[dict]` — pure; valid ops are `{"op": "add", "content": str}`, `{"op": "update", "id": str, "content": str}`, `{"op": "delete", "id": str}`; malformed input → `[]`; max 3 ops.
  - `MAX_MEMORY_OPS_PER_TURN = 3`, `MEMORY_EXTRACTION_TIMEOUT_SECONDS = 15`, `MAX_MEMORY_CONTENT_LENGTH = 500`.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for user_memory_service pure functions."""

import pytest

from app.services.user_memory_service import (
    MAX_MEMORY_OPS_PER_TURN,
    build_memory_block,
    parse_memory_ops,
)


class FakeMemory:
    def __init__(self, content):
        self.content = content


def test_build_memory_block_empty():
    assert build_memory_block([]) == ""


def test_build_memory_block_formats_facts():
    block = build_memory_block([FakeMemory("Prefers concise answers"), FakeMemory("Name is Alex")])
    assert block.startswith("About the user (persistent memory; use when relevant):")
    assert "- Prefers concise answers" in block
    assert "- Name is Alex" in block


def test_parse_memory_ops_valid_json():
    raw = '[{"op": "add", "content": "Likes Rust"}]'
    assert parse_memory_ops(raw) == [{"op": "add", "content": "Likes Rust"}]


def test_parse_memory_ops_strips_code_fence():
    raw = '```json\n[{"op": "delete", "id": "abc"}]\n```'
    assert parse_memory_ops(raw) == [{"op": "delete", "id": "abc"}]


def test_parse_memory_ops_invalid_json_returns_empty():
    assert parse_memory_ops("not json at all") == []
    assert parse_memory_ops("") == []


def test_parse_memory_ops_rejects_bad_shapes():
    raw = """
    [
      {"op": "add"},
      {"op": "add", "content": ""},
      {"op": "bogus", "content": "x"},
      {"op": "update", "content": "no id"},
      {"op": "delete", "id": 42},
      {"op": "add", "content": 7},
      "just a string",
      {"op": "add", "content": "Valid one"}
    ]
    """
    assert parse_memory_ops(raw) == [{"op": "add", "content": "Valid one"}]


def test_parse_memory_ops_caps_at_three():
    raw = "[" + ",".join(f'{{"op": "add", "content": "c{i}"}}' for i in range(10)) + "]"
    assert len(parse_memory_ops(raw)) == MAX_MEMORY_OPS_PER_TURN


def test_parse_memory_ops_truncates_overlong_content():
    raw = '[{"op": "add", "content": "' + "x" * 600 + '"}]'
    ops = parse_memory_ops(raw)
    assert len(ops[0]["content"]) == 500
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_user_memory_service.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the pure functions**

```python
"""
User-level persistent memory: prompt-block formatting, extraction-op parsing,
extraction via the LLM gateway, and the fire-and-forget update orchestration.

All LLM work goes through ``app.services.llm_gateway`` using the request's own
backend/model config, so behavior is identical on Ollama, OpenRouter, and
Nvidia NIM. Extraction failures are logged and skipped — they never affect the
user-visible response.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from app.services.llm_gateway import complete_text

logger = logging.getLogger(__name__)

MAX_MEMORY_OPS_PER_TURN = 3
MEMORY_EXTRACTION_TIMEOUT_SECONDS = 15
MAX_MEMORY_CONTENT_LENGTH = 500

MEMORY_BLOCK_HEADER = "About the user (persistent memory; use when relevant):"


def build_memory_block(memories: List[Any]) -> str:
    """Format memory records into a system-prompt block ("" when none)."""
    contents = [getattr(m, "content", "") for m in memories if getattr(m, "content", "")]
    if not contents:
        return ""
    lines = [MEMORY_BLOCK_HEADER] + [f"- {c}" for c in contents]
    return "\n".join(lines)


def _valid_op(item: Any) -> Optional[Dict[str, Any]]:
    """Return a normalized op dict, or None if the item is malformed."""
    if not isinstance(item, dict):
        return None
    op = item.get("op")
    if op == "add":
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            return {"op": "add", "content": content.strip()[:MAX_MEMORY_CONTENT_LENGTH]}
        return None
    if op in ("update", "delete"):
        memory_id = item.get("id")
        if not isinstance(memory_id, str) or not memory_id:
            return None
        if op == "delete":
            return {"op": "delete", "id": memory_id}
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            return {"op": "update", "id": memory_id, "content": content.strip()[:MAX_MEMORY_CONTENT_LENGTH]}
        return None
    return None


def parse_memory_ops(raw: str) -> List[Dict[str, Any]]:
    """Parse the extraction LLM's output into validated ops (max 3)."""
    if not raw or not raw.strip():
        return []
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.debug("Memory extraction returned non-JSON output, skipping")
        return []
    if not isinstance(data, list):
        return []
    ops = []
    for item in data:
        op = _valid_op(item)
        if op is not None:
            ops.append(op)
        if len(ops) >= MAX_MEMORY_OPS_PER_TURN:
            break
    return ops
```

Note: `text.strip("`")` also strips backticks inside the fence marker — acceptable because the input is model output that either is a fenced block or raw JSON; the JSON parse afterwards is the real validator.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_user_memory_service.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check app/services/user_memory_service.py tests/test_user_memory_service.py
uv run ty check app/services
git add app/services/user_memory_service.py tests/test_user_memory_service.py
git commit -m "feat: memory block formatting and extraction op parsing"
```

---

### Task 3: Extraction + orchestration (gateway-backed)

**Files:**
- Modify: `app/services/user_memory_service.py` (append)
- Test: `tests/test_user_memory_service.py` (append)

**Interfaces:**
- Consumes: `complete_text(messages, **kwargs) -> str` from `app.services.llm_gateway` (streams internally, returns full text); `parse_memory_ops`, `build_memory_block` from Task 2; `get_user_memory_store()` from Task 1.
- Produces:
  - `async def extract_memory_ops(user_message: str, assistant_message: str, existing: List[Any], *, backend=None, model=None, api_base=None, api_key=None) -> List[dict]` — validated ops (may be `[]`); raises nothing (returns `[]` on timeout/error).
  - `async def apply_memory_ops(store, user_id: str, ops: List[dict]) -> None`.
  - `async def run_memory_update(user_id: str, user_message: str, assistant_message: str, *, backend=None, model=None, api_base=None, api_key=None) -> None` — full pipeline, swallows all exceptions (fire-and-forget safe).
  - `async def load_memory_block(user_id: Optional[str]) -> str` — store read + `build_memory_block`; `""` on any failure or when `user_id` is None.

- [ ] **Step 1: Write the failing tests**

```python
# ── Extraction and orchestration (Task 3) ──────────────────────────────────

import asyncio
from unittest.mock import AsyncMock, patch

import app.services.user_memory_service as ums
from app.services.user_memory_service import (
    apply_memory_ops,
    extract_memory_ops,
    load_memory_block,
    run_memory_update,
)


class FakeStore:
    def __init__(self, memories=None):
        self.memories = list(memories or [])
        self.added = []
        self.updated = []
        self.deleted = []

    async def list_memories(self, user_id, limit=50):
        return self.memories

    async def add_memory(self, user_id, content, source="auto"):
        self.added.append((user_id, content, source))

        class R:
            content = content

        return R()

    async def update_memory(self, user_id, memory_id, content):
        self.updated.append((user_id, memory_id, content))
        return True

    async def delete_memory(self, user_id, memory_id):
        self.deleted.append((user_id, memory_id))
        return True


@pytest.mark.asyncio
async def test_extract_memory_ops_sends_prompt_and_parses():
    with patch.object(ums, "complete_text", new=AsyncMock(return_value='[{"op": "add", "content": "Owns a dog"}]')) as mock_llm:
        ops = await extract_memory_ops("I own a dog named Rex", "That's lovely!", [], backend="ollama", model="gemma3:1b")
    assert ops == [{"op": "add", "content": "Owns a dog"}]
    sent = mock_llm.call_args.args[0]
    assert any(m["role"] == "system" for m in sent)
    assert "I own a dog named Rex" in sent[-1]["content"]
    assert mock_llm.call_args.kwargs["backend"] == "ollama"


@pytest.mark.asyncio
async def test_extract_memory_ops_returns_empty_on_llm_error():
    with patch.object(ums, "complete_text", new=AsyncMock(side_effect=RuntimeError("provider down"))):
        assert await extract_memory_ops("hi", "hello", []) == []


@pytest.mark.asyncio
async def test_apply_memory_ops_dispatches_to_store():
    store = FakeStore()
    await apply_memory_ops(
        store, "user-1",
        [
            {"op": "add", "content": "New fact"},
            {"op": "update", "id": "m1", "content": "Edited"},
            {"op": "delete", "id": "m2"},
        ],
    )
    assert store.added == [("user-1", "New fact", "auto")]
    assert store.updated == [("user-1", "m1", "Edited")]
    assert store.deleted == [("user-1", "m2")]


@pytest.mark.asyncio
async def test_run_memory_update_swallows_all_errors():
    with patch.object(ums, "get_user_memory_store", side_effect=RuntimeError("db down")):
        await run_memory_update("user-1", "hi", "hello")  # must not raise


@pytest.mark.asyncio
async def test_run_memory_update_happy_path():
    store = FakeStore(memories=[])
    with patch.object(ums, "get_user_memory_store", return_value=store), \
         patch.object(ums, "complete_text", new=AsyncMock(return_value='[{"op": "add", "content": "Lives in Berlin"}]')):
        await run_memory_update("user-1", "I live in Berlin", "Nice city!")
    assert store.added == [("user-1", "Lives in Berlin", "auto")]


@pytest.mark.asyncio
async def test_load_memory_block_empty_user():
    assert await load_memory_block(None) == ""


@pytest.mark.asyncio
async def test_load_memory_block_degrades_on_store_error():
    with patch.object(ums, "get_user_memory_store", side_effect=RuntimeError("db down")):
        assert await load_memory_block("user-1") == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_user_memory_service.py -v`
Expected: FAIL — `ImportError: cannot import name 'extract_memory_ops'`

- [ ] **Step 3: Implement extraction and orchestration**

Append to `app/services/user_memory_service.py`:

```python
_EXTRACTION_SYSTEM_PROMPT = """You maintain long-term memory for a personal assistant. \
Given a conversation exchange and the user's existing memories, decide what durable facts about the \
user should be stored, updated, or removed.

Return ONLY a JSON array. Each element is exactly one of:
{"op": "add", "content": "<new memory, one concise sentence>"}
{"op": "update", "id": "<existing memory id>", "content": "<corrected memory>"}
{"op": "delete", "id": "<existing memory id that is now wrong or obsolete>"}

Rules:
- Store only durable facts: the user's identity, stable preferences, ongoing projects or goals, \
corrections to existing memories, and explicit requests to remember something.
- Never store secrets, credentials, API keys, passwords, or transient task details.
- Prefer updating an existing memory over adding a near-duplicate.
- At most 3 operations. Return [] when nothing durable was shared."""


def _build_extraction_messages(
    user_message: str, assistant_message: str, existing: List[Any]
) -> List[Dict[str, str]]:
    existing_lines = "\n".join(
        f'- id={getattr(m, "id", "?")}: {getattr(m, "content", "")}' for m in existing
    )
    return [
        {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Existing memories:\n{existing_lines or '(none)'}\n\n"
                f"User said:\n{user_message}\n\n"
                f"Assistant replied:\n{assistant_message}"
            ),
        },
    ]


async def extract_memory_ops(
    user_message: str,
    assistant_message: str,
    existing: List[Any],
    *,
    backend: Optional[str] = None,
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Ask the LLM which memories to add/update/delete; [] on any failure."""
    if not user_message.strip():
        return []
    messages = _build_extraction_messages(user_message, assistant_message, existing)
    try:
        raw = await asyncio.wait_for(
            complete_text(
                messages,
                backend=backend,
                model=model,
                api_base=api_base,
                api_key=api_key,
                max_tokens=300,
            ),
            timeout=MEMORY_EXTRACTION_TIMEOUT_SECONDS,
        )
    except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001 — extraction must never raise
        logger.warning("Memory extraction failed/skipped: %s", exc)
        return []
    return parse_memory_ops(raw)


async def apply_memory_ops(store: Any, user_id: str, ops: List[Dict[str, Any]]) -> None:
    """Apply validated ops to the store; unknown ids are ignored by the store."""
    for op in ops:
        if op["op"] == "add":
            await store.add_memory(user_id, op["content"], source="auto")
        elif op["op"] == "update":
            await store.update_memory(user_id, op["id"], op["content"])
        elif op["op"] == "delete":
            await store.delete_memory(user_id, op["id"])


async def run_memory_update(
    user_id: str,
    user_message: str,
    assistant_message: str,
    *,
    backend: Optional[str] = None,
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
) -> None:
    """Extract and apply memory ops for one completed exchange. Never raises."""
    try:
        store = get_user_memory_store()
        existing = await store.list_memories(user_id)
        ops = await extract_memory_ops(
            user_message,
            assistant_message,
            existing,
            backend=backend,
            model=model,
            api_base=api_base,
            api_key=api_key,
        )
        await apply_memory_ops(store, user_id, ops)
        if ops:
            logger.info("User memory updated for user %s (%d ops)", user_id, len(ops))
    except Exception as exc:  # noqa: BLE001 — fire-and-forget task must be self-contained
        logger.warning("User memory update failed for user %s: %s", user_id, exc)


async def load_memory_block(user_id: Optional[str]) -> str:
    """Load and format the user's memories for prompt injection ("" on failure)."""
    if not user_id:
        return ""
    try:
        store = get_user_memory_store()
        memories = await store.list_memories(user_id)
        return build_memory_block(memories)
    except Exception as exc:  # noqa: BLE001 — degraded response beats failed response
        logger.warning("Failed to load user memory for user %s: %s", user_id, exc)
        return ""
```

Add this import at the top of the file (merge with existing imports):

```python
from app.backend.user_memory_store import get_user_memory_store
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_user_memory_service.py -v`
Expected: PASS (15 tests)

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check app/services/user_memory_service.py tests/test_user_memory_service.py
uv run ty check app/services
git add app/services/user_memory_service.py tests/test_user_memory_service.py
git commit -m "feat: LLM-backed memory extraction and fire-and-forget update"
```

---

### Task 4: `/api/memory` CRUD router

**Files:**
- Modify: `app/backend/models/api_models.py` (append models)
- Create: `app/backend/api/memory.py`
- Modify: `main.py` (router registration — after `app.include_router(voice_api.router)`, around `main.py:343`)
- Test: `tests/test_memory_api.py`

**Interfaces:**
- Consumes: `get_current_user`/`User` from `app.backend.api.auth_deps`; `get_user_memory_store` from Task 1; store methods from Task 1.
- Produces: routes `GET /api/memory`, `POST /api/memory` (body `{"content": str}` → source `"manual"`), `PUT /api/memory/{memory_id}` (body `{"content": str}`), `DELETE /api/memory/{memory_id}`, `POST /api/memory/clear`. Pydantic models `MemoryContentRequest(content: str)`, `MemoryRecordResponse(id: str, content: str, source: str, created_at: str, updated_at: str)` in `api_models.py`.

- [ ] **Step 1: Write the failing tests**

```python
"""API tests for /api/memory — auth required, user-scoped CRUD."""

import uuid

import pytest
from fastapi.testclient import TestClient


class FakeStore:
    def __init__(self):
        self.rows = {}

    async def list_memories(self, user_id, limit=50):
        return [r for r in self.rows.values() if r.user_id == user_id]

    async def add_memory(self, user_id, content, source="auto"):
        from app.backend.user_memory_store import UserMemoryRecord

        rec = UserMemoryRecord(
            id=str(uuid.uuid4()), user_id=user_id, content=content, source=source,
            created_at="2026-08-23T00:00:00", updated_at="2026-08-23T00:00:00",
        )
        self.rows[rec.id] = rec
        return rec

    async def update_memory(self, user_id, memory_id, content):
        r = self.rows.get(memory_id)
        if not r or r.user_id != user_id:
            return False
        r.content = content
        return True

    async def delete_memory(self, user_id, memory_id):
        r = self.rows.get(memory_id)
        if not r or r.user_id != user_id:
            return False
        del self.rows[memory_id]
        return True

    async def clear_memories(self, user_id):
        ids = [i for i, r in self.rows.items() if r.user_id == user_id]
        for i in ids:
            del self.rows[i]
        return len(ids)


@pytest.fixture()
def client(monkeypatch):
    from fastapi.testclient import TestClient
    from main import app
    import app.backend.api.memory as memory_api
    from app.backend.api.auth_deps import User, get_current_user

    store = FakeStore()
    # Router resolves the singleton via its own module namespace at call time,
    # so patching memory_api.get_user_memory_store intercepts every endpoint.
    monkeypatch.setattr(memory_api, "get_user_memory_store", lambda: store)
    # Same override pattern as tests/test_chat_auth.py:_make_app
    app.dependency_overrides[get_current_user] = lambda: User(
        id=str(uuid.uuid4()), email="u1@x.com"
    )
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.pop(get_current_user, None)


def test_memory_requires_auth():
    from fastapi.testclient import TestClient
    from main import app

    with TestClient(app) as c:
        assert c.get("/api/memory").status_code == 401


def test_memory_crud_roundtrip(client):
    resp = client.post("/api/memory", json={"content": "Prefers dark mode"})
    assert resp.status_code == 200
    mid = resp.json()["id"]
    assert resp.json()["source"] == "manual"

    listed = client.get("/api/memory").json()
    assert [m["content"] for m in listed] == ["Prefers dark mode"]

    assert client.put(f"/api/memory/{mid}", json={"content": "Prefers light mode"}).status_code == 200
    assert client.get("/api/memory").json()[0]["content"] == "Prefers light mode"

    assert client.delete(f"/api/memory/{mid}").status_code == 200
    assert client.get("/api/memory").json() == []


def test_memory_add_rejects_blank_content(client):
    assert client.post("/api/memory", json={"content": "   "}).status_code == 422


def test_memory_update_unknown_id_404(client):
    assert client.put(f"/api/memory/{uuid.uuid4()}", json={"content": "x"}).status_code == 404


def test_memory_clear(client):
    client.post("/api/memory", json={"content": "a"})
    client.post("/api/memory", json={"content": "b"})
    resp = client.post("/api/memory/clear")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == 2
    assert client.get("/api/memory").json() == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_memory_api.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.backend.api.memory'`

- [ ] **Step 3: Add models, router, registration**

Append to `app/backend/models/api_models.py`:

```python
class MemoryContentRequest(BaseModel):
    content: str = Field(..., min_length=1, description="Memory text (trimmed server-side)")


class MemoryRecordResponse(BaseModel):
    id: str
    content: str
    source: str
    created_at: str
    updated_at: str
```

(If `Field` is not yet imported in that file, add it to the existing `pydantic` import.)

Create `app/backend/api/memory.py`:

```python
"""User-level persistent memory CRUD endpoints (all require authentication)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.backend.api.auth_deps import User, get_current_user
from app.backend.models.api_models import MemoryContentRequest, MemoryRecordResponse
from app.backend.user_memory_store import get_user_memory_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/memory", tags=["memory"])


def _response(record) -> MemoryRecordResponse:
    return MemoryRecordResponse(
        id=record.id,
        content=record.content,
        source=record.source,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


@router.get("", response_model=list[MemoryRecordResponse])
async def list_memories(user: User = Depends(get_current_user)):
    records = await get_user_memory_store().list_memories(user.id)
    return [_response(r) for r in records]


@router.post("", response_model=MemoryRecordResponse)
async def add_memory(request: MemoryContentRequest, user: User = Depends(get_current_user)):
    content = request.content.strip()
    if not content:
        raise HTTPException(status_code=422, detail="Memory content cannot be empty")
    record = await get_user_memory_store().add_memory(user.id, content, source="manual")
    return _response(record)


@router.put("/{memory_id}")
async def update_memory(memory_id: str, request: MemoryContentRequest, user: User = Depends(get_current_user)):
    content = request.content.strip()
    if not content:
        raise HTTPException(status_code=422, detail="Memory content cannot be empty")
    updated = await get_user_memory_store().update_memory(user.id, memory_id, content)
    if not updated:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"status": "updated", "id": memory_id}


@router.delete("/{memory_id}")
async def delete_memory(memory_id: str, user: User = Depends(get_current_user)):
    deleted = await get_user_memory_store().delete_memory(user.id, memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"status": "deleted", "id": memory_id}


@router.post("/clear")
async def clear_memories(user: User = Depends(get_current_user)):
    count = await get_user_memory_store().clear_memories(user.id)
    return {"status": "cleared", "deleted": count}
```

In `main.py`, add the import next to the other API imports (around `main.py:27-30`):

```python
from app.backend.api import memory as memory_api
```

and register after `app.include_router(voice_api.router)` (`main.py:343`):

```python
app.include_router(memory_api.router)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_memory_api.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check app/backend/api/memory.py main.py tests/test_memory_api.py
uv run ty check app/backend main.py
git add app/backend/api/memory.py app/backend/models/api_models.py main.py tests/test_memory_api.py
git commit -m "feat: /api/memory CRUD endpoints"
```

---

### Task 5: Chat injection + extraction triggers

**Files:**
- Modify: `app/backend/api/chat.py` — `_build_messages_with_history` (line 159), `_handle_chat_request` (278), `_stream_chat_response` (311), `chat_endpoint` (246), `chat_stream` (259)
- Test: `tests/test_user_memory_wiring.py`

**Interfaces:**
- Consumes: `load_memory_block(user_id) -> str`, `run_memory_update(user_id, user_message, assistant_message, *, backend, model, api_base, api_key)` from Task 3.
- Produces: `_build_messages_with_history(request, memory_block: Optional[str] = None) -> List[Dict[str, str]]`; `_handle_chat_request(request, user_id: Optional[str] = None)`; `_stream_chat_response(request, user_id: Optional[str] = None)`. All chat routes pass `user.id`.

- [ ] **Step 1: Write the failing tests**

```python
"""Chat wiring: memory block injection + post-exchange extraction trigger."""

import uuid
from unittest.mock import AsyncMock, patch

import pytest

import app.backend.api.chat as chat_api
from app.backend.models.api_models import ChatMessage, ChatRequestEnhanced


def _request(**overrides):
    base = dict(message="Remember I like tea", backend="ollama", model="gemma3:1b")
    base.update(overrides)
    return ChatRequestEnhanced(**base)


@pytest.mark.asyncio
async def test_stream_injects_memory_block():
    with patch.object(chat_api, "load_memory_block", new=AsyncMock(return_value="About the user:\n- Likes tea")), \
         patch.object(chat_api, "stream_completion", new=AsyncMock(return_value=aiter(["Hello"]))):
        chunks = [c async for c in chat_api._stream_chat_response(_request(), user_id="u-1")]
    assert chunks == ["Hello"]
    sent = chat_api.stream_completion.call_args.args[0]
    assert sent[0] == {"role": "system", "content": "About the user:\n- Likes tea"}


@pytest.mark.asyncio
async def test_stream_skips_block_when_empty():
    with patch.object(chat_api, "load_memory_block", new=AsyncMock(return_value="")), \
         patch.object(chat_api, "stream_completion", new=AsyncMock(return_value=aiter(["Hi"]))):
        await anext(chat_api._stream_chat_response(_request(), user_id="u-1"))
    sent = chat_api.stream_completion.call_args.args[0]
    assert sent[0]["role"] != "system" or "persistent memory" not in sent[0]["content"]


@pytest.mark.asyncio
async def test_non_stream_path_triggers_extraction():
    with patch.object(chat_api, "load_memory_block", new=AsyncMock(return_value="")), \
         patch.object(chat_api, "complete_text", new=AsyncMock(return_value="Enjoy your tea!")), \
         patch.object(chat_api.asyncio, "create_task") as mock_task, \
         patch.object(chat_api, "run_memory_update", new=AsyncMock()):
        await chat_api._handle_chat_request(_request(), user_id="u-1")
    assert mock_task.called


@pytest.mark.asyncio
async def test_handle_chat_request_without_user_skips_memory_load():
    with patch.object(chat_api, "load_memory_block", new=AsyncMock(return_value="")) as mock_load, \
         patch.object(chat_api, "complete_text", new=AsyncMock(return_value="ok")):
        await chat_api._handle_chat_request(_request())
    mock_load.assert_not_called()


async def aiter(items):
    for i in items:
        yield i
```

**Note:** if `test_chat_auth.py` already patches gateway functions differently (e.g. patching `stream_completion` on the gateway module), match its existing patch targets so both suites stay green — read `tests/test_chat_auth.py` before finalizing the mocks above.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_user_memory_wiring.py -v`
Expected: FAIL — `_stream_chat_response() got an unexpected keyword argument 'user_id'`

- [ ] **Step 3: Implement the wiring**

In `app/backend/api/chat.py`:

Add imports near the top:

```python
from app.services.user_memory_service import load_memory_block, run_memory_update
```

Change `_build_messages_with_history` signature and prepend the block (keep the rest of the function identical):

```python
def _build_messages_with_history(
    request: ChatRequestEnhanced, memory_block: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Build the messages list including conversation history and summary.
    Also applies voice mode optimizations if is_voice_mode is True.

    Args:
        request: The chat request with optional history/summary
        memory_block: Pre-formatted persistent user-memory block (may be empty)

    Returns:
        List of messages ready for LLM
    """
    messages = []

    # Persistent user memory comes first so later system prompts can refine it
    if memory_block:
        messages.append({"role": "system", "content": memory_block})

    # Add voice mode system prompt if voice mode is active
    if request.is_voice_mode:
        ...  # existing body unchanged from here down
```

Change the two callers to load and pass the block:

```python
async def _handle_chat_request(request: ChatRequestEnhanced, user_id: Optional[str] = None) -> ChatResponse:
    """Handle a chat request with conversation history."""
    memory_block = await load_memory_block(user_id) if user_id else ""
    messages = _build_messages_with_history(request, memory_block=memory_block)
    # ... existing body unchanged ...
    return ChatResponse(response=response_text, model=request.model if request.model is not None else "default")
```

```python
async def _stream_chat_response(request: ChatRequestEnhanced, user_id: Optional[str] = None):
    """Stream chat responses with conversation history."""
    memory_block = await load_memory_block(user_id) if user_id else ""
    messages = _build_messages_with_history(request, memory_block=memory_block)
    # ... existing body unchanged ...
```

Update `chat_endpoint` to trigger extraction after a successful response:

```python
@router.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequestEnhanced, user: User = Depends(get_current_user)):
    """Single chat message processing (requires authentication)."""
    logger.info(f"Chat request - User: {user.id}, Backend: {request.backend}, Model: {request.model}")

    try:
        response = await _handle_chat_request(request, user_id=user.id)
        asyncio.create_task(
            run_memory_update(
                user.id,
                request.message,
                response.response,
                backend=request.backend,
                model=request.model,
                api_base=request.api_base,
                api_key=request.api_key,
            )
        )
        return response

    except Exception:
        logger.exception("Chat endpoint error")
        raise HTTPException(status_code=500, detail="An internal error occurred")
```

Update `chat_stream` to accumulate and trigger after a clean stream:

```python
@router.post("/api/chat/stream")
async def chat_stream(request: ChatRequestEnhanced, user: User = Depends(get_current_user)):
    """Stream chat responses (requires authentication)."""
    logger.info(f"Streaming chat - User: {user.id}, Backend: {request.backend}, Model: {request.model}")

    async def event_generator():
        collected: List[str] = []
        try:
            async for chunk in _stream_chat_response(request, user_id=user.id):
                collected.append(chunk)
                payload = json.dumps({"chunk": chunk}, ensure_ascii=False)
                yield f"data: {payload}\n\n"

            if collected:
                asyncio.create_task(
                    run_memory_update(
                        user.id,
                        request.message,
                        "".join(collected),
                        backend=request.backend,
                        model=request.model,
                        api_base=request.api_base,
                        api_key=request.api_key,
                    )
                )

        except Exception:
            logger.exception("Chat streaming error")
            payload = json.dumps({"error": "An internal error occurred"}, ensure_ascii=False)
            yield f"data: {payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

(Extraction is intentionally skipped when the stream errors — partial exchanges produce noisy memories.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_user_memory_wiring.py tests/test_chat_auth.py -v`
Expected: PASS — new tests and existing chat auth tests

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check app/backend/api/chat.py tests/test_user_memory_wiring.py
uv run ty check app/backend
git add app/backend/api/chat.py tests/test_user_memory_wiring.py
git commit -m "feat: inject user memory into chat and extract after exchanges"
```

---

### Task 6: RAG injection

**Files:**
- Modify: `app/services/rag_service.py` — `query(...)` (starts `app/services/rag_service.py:1543`; message assembly at ~1615-1652)
- Modify: `app/skills/rag.py` — both skills' `stream()` (StandardRAGSkill at `app/skills/rag.py:26-52`, MultiAgentRAGSkill at `app/skills/rag.py:66-95`)
- Modify: `main.py` — `rag_query` (`main.py:765-785`)
- Test: `tests/test_user_memory_wiring.py` (append)

**Interfaces:**
- Consumes: `load_memory_block` from Task 3; `rag_skill_registry` skills' `stream(request, rag_service)` protocol (extended additively with a default-valued third parameter).
- Produces: `StandardRAGSkill.stream(request, rag_service, memory_block: Optional[str] = None)`; `MultiAgentRAGSkill.stream(..., memory_block=None)` (forwards to fallback skill; multi-agent orchestrator path ignores it — CrewAI path unchanged); `rag_service.query(..., memory_block: Optional[str] = None)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_user_memory_wiring.py`:

```python
# ── RAG injection ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rag_query_injects_memory_block():
    """query(..., memory_block=...) must place the block before conversation history."""
    from app.services import rag_service as rag_module
    from app.services.rag_service import RAGService

    svc = RAGService.__new__(RAGService)  # retrieval + prompt seams are patched below

    block = {"role": "system", "content": "About the user:\n- Likes tea"}
    with patch.object(svc, "get_retrieval_records", return_value=[]), \
         patch.object(svc, "build_grounded_user_prompt", return_value="grounded"), \
         patch.object(rag_module, "stream_completion", new=AsyncMock(return_value=aiter(["answer"]))):
        chunks = [
            c
            async for c in svc.query(
                "what is x",
                system_prompt="You are helpful.",
                messages=[{"role": "user", "content": "hi"}],
                memory_block="About the user:\n- Likes tea",
            )
        ]

    assert "answer" in chunks
    sent = rag_module.stream_completion.call_args.args[0]
    assert block in sent
    assert sent.index(block) < sent.index({"role": "user", "content": "hi"})
```

**Verified seams** (`app/services/rag_service.py`, `query()` body): retrieval goes through `self.get_retrieval_records(full_query, n_results, use_hybrid_search)` (~line 1588); the final LLM call is the module-level `stream_completion(llm_messages, ...)` imported at line 25 — hence the `rag_module.stream_completion` patch target. The empty-retrieval path is safe: score normalization handles `[]`, and a citations JSON line is yielded before the answer chunks.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_user_memory_wiring.py -v`
Expected: FAIL — `query() got an unexpected keyword argument 'memory_block'`

- [ ] **Step 3: Implement**

In `app/services/rag_service.py`, add the parameter to `query(...)` (after `is_voice_mode: bool = False,`):

```python
            memory_block: Optional[str] = None,
```

and inject it in the message assembly (immediately after the conversation-summary append at ~1617-1621):

```python
            # Persistent user memory (may be empty)
            if memory_block:
                llm_messages.append({"role": "system", "content": memory_block})
```

In `app/skills/rag.py`, change both signatures and forwarding:

```python
    async def stream(
        self, request: Any, rag_service: Any, memory_block: Optional[str] = None
    ) -> AsyncIterator[str]:
        async for chunk in rag_service.query(
            request.query,
            request.system_prompt,
            _rag_messages(request),
            request.n_results,
            request.use_hybrid_search,
            request.model,
            conversation_summary=request.conversation_summary,
            memory_block=memory_block,
            backend=request.backend,
            # ... remaining kwargs unchanged ...
        ):
            yield chunk
```

(Add `from typing import Optional` to the imports if absent.)

For `MultiAgentRAGSkill.stream`, add the same `memory_block: Optional[str] = None` parameter and forward it in **both** fallback calls:

```python
            async for chunk in self._fallback_skill.stream(request, rag_service, memory_block=memory_block):
                yield chunk
```

(The CrewAI `orchestrator.query(...)` call stays unchanged — multi-agent mode does not inject memory v1.)

In `main.py` `rag_query` (`main.py:765`), load the block and pass it through:

```python
@app.post("/api/rag/query")
async def rag_query(request: RAGQueryRequestEnhanced, user: User = Depends(get_current_user)):
    """Query RAG system (requires authentication)."""
    try:
        if not rag_service:
            raise HTTPException(status_code=503, detail="RAG service not initialized")

        skill = rag_skill_registry.resolve(request)
        if skill is None:
            raise HTTPException(status_code=400, detail="No compatible RAG skill found for request")

        from app.services.user_memory_service import load_memory_block

        memory_block = await load_memory_block(user.id)

        async def event_generator():
            logger.info("Routing RAG query through skill '%s'", skill.name)
            async for chunk in skill.stream(request, rag_service, memory_block=memory_block):
                yield chunk + "\n"

        return StreamingResponse(event_generator(), media_type="text/plain")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_user_memory_wiring.py tests/test_rag_skills.py -v`
Expected: PASS — new test and existing RAG skill tests

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check app/services/rag_service.py app/skills/rag.py main.py tests/test_user_memory_wiring.py
uv run ty check app/services app/skills main.py
git add app/services/rag_service.py app/skills/rag.py main.py tests/test_user_memory_wiring.py
git commit -m "feat: inject user memory into RAG answers"
```

---

### Task 7: Voice pipeline injection + per-turn extraction

**Files:**
- Modify: `app/services/voice_pipeline.py` — `build_voice_pipeline` (line 73) and `BackendLLMService._process_context` (line 160)
- Test: `tests/test_user_memory_wiring.py` (append)

**Interfaces:**
- Consumes: `VoiceSettings(user_id, system_instruction, backend, model, api_key, api_base)` (already exists); `load_memory_block`, `run_memory_update` from Task 3.
- Produces: `build_voice_pipeline` becomes aware of memory (still sync-callable? No — it must become `async` if it awaits; check its caller `run_voice_pipeline`/`voice.py` and make it `async def` + await at the call site, or load memories in `run_voice_pipeline` before calling it — prefer loading in `run_voice_pipeline` and passing `memory_block` into `VoiceSettings.system_instruction` mutation to keep `build_voice_pipeline` sync).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_user_memory_wiring.py`:

```python
# ── Voice pipeline ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_voice_pipeline_prepends_memory_to_system_instruction():
    from app.services.voice_pipeline import VoiceSettings

    settings = VoiceSettings(user_id="u-1", system_instruction="You are a helpful voice assistant.")
    with patch("app.services.voice_pipeline.load_memory_block", new=AsyncMock(return_value="About the user:\n- Likes tea")):
        from app.services.voice_pipeline import apply_memory_to_voice_settings

        await apply_memory_to_voice_settings(settings)
    assert settings.system_instruction.startswith("You are a helpful voice assistant.")
    assert "Likes tea" in settings.system_instruction


@pytest.mark.asyncio
async def test_voice_pipeline_skips_memory_without_user():
    from app.services.voice_pipeline import VoiceSettings, apply_memory_to_voice_settings

    settings = VoiceSettings(user_id=None, system_instruction="base")
    with patch("app.services.voice_pipeline.load_memory_block", new=AsyncMock(return_value="X")) as m:
        await apply_memory_to_voice_settings(settings)
    m.assert_not_called()
    assert settings.system_instruction == "base"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_user_memory_wiring.py -v`
Expected: FAIL — `ImportError: cannot import name 'apply_memory_to_voice_settings'`

- [ ] **Step 3: Implement**

In `app/services/voice_pipeline.py`:

Add import:

```python
from app.services.user_memory_service import load_memory_block, run_memory_update
```

Add helper next to `VoiceSettings`:

```python
async def apply_memory_to_voice_settings(settings: VoiceSettings) -> None:
    """Fold the user's persistent memory into the voice system instruction."""
    if not settings.user_id:
        return
    block = await load_memory_block(settings.user_id)
    if block:
        settings.system_instruction = f"{settings.system_instruction}\n\n{block}"
```

In `run_voice_pipeline` (line 254), before building the pipeline:

```python
    await apply_memory_to_voice_settings(settings)
```

In `BackendLLMService._process_context` (line 160), capture the exchange and schedule extraction after the stream completes:

```python
    async def _process_context(self, context: LLMContext):
        raw_messages = cast("list[dict[str, Any]]", context.get_messages())
        messages = [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in raw_messages
        ]
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        reply_parts: list[str] = []
        async for delta in stream_completion(
            messages,
            backend=self._voice_settings.backend,
            model=self._voice_settings.model,
            api_key=self._voice_settings.api_key,
            api_base=self._voice_settings.api_base,
            temperature=self._voice_settings.temperature,
            max_tokens=self._voice_settings.max_tokens,
        ):
            if delta:
                reply_parts.append(delta)
                await self.push_frame(LLMTextFrame(text=delta))

        # Fire-and-forget memory extraction for this voice turn
        if self._voice_settings.user_id and last_user and reply_parts:
            asyncio.create_task(
                run_memory_update(
                    self._voice_settings.user_id,
                    last_user,
                    "".join(reply_parts),
                    backend=self._voice_settings.backend,
                    model=self._voice_settings.model,
                    api_base=self._voice_settings.api_base,
                    api_key=self._voice_settings.api_key,
                )
            )
```

(Add `import asyncio` at the top if absent.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_user_memory_wiring.py tests/test_rag_voice_auth.py -v`
Expected: PASS

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check app/services/voice_pipeline.py tests/test_user_memory_wiring.py
uv run ty check app/services
git add app/services/voice_pipeline.py tests/test_user_memory_wiring.py
git commit -m "feat: user memory in voice mode (injection + per-turn extraction)"
```

---

### Task 8: Frontend — memory API client + Settings panel section

**Files:**
- Modify: `nextjs-frontend/src/lib/types.ts` (append type)
- Modify: `nextjs-frontend/src/lib/api.ts` (append client functions)
- Modify: `nextjs-frontend/src/components/settings-panel.tsx` (add Memory section — follow the existing `Section` component at the top of the file, used at e.g. line 181/407/430)
- Test: `nextjs-frontend/src/components/memory-settings.test.tsx`

**Interfaces:**
- Consumes: `/api/memory` endpoints from Task 4; existing `API_BASE`/`credentials: 'include'` fetch pattern in `api.ts`.
- Produces: `MemoryRecord` type; `getMemories()`, `addMemory(content)`, `updateMemory(id, content)`, `deleteMemory(id)`, `clearMemories()`.

- [ ] **Step 1: Write the failing test**

`nextjs-frontend/src/components/memory-settings.test.tsx`:

```tsx
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemorySettings } from './settings-panel';
import * as api from '@/lib/api';

vi.mock('@/lib/api', () => ({
  getMemories: vi.fn(),
  addMemory: vi.fn(),
  updateMemory: vi.fn(),
  deleteMemory: vi.fn(),
  clearMemories: vi.fn(),
}));

const mem = (id: string, content: string) => ({
  id, content, source: 'auto', created_at: '2026-08-23T00:00:00', updated_at: '2026-08-23T00:00:00',
});

describe('MemorySettings', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.getMemories).mockResolvedValue([mem('m1', 'Prefers dark mode')]);
  });

  it('lists memories on load', async () => {
    render(<MemorySettings />);
    expect(await screen.findByText('Prefers dark mode')).toBeInTheDocument();
  });

  it('adds a memory', async () => {
    vi.mocked(api.addMemory).mockResolvedValue(mem('m2', 'New fact'));
    render(<MemorySettings />);
    await screen.findByText('Prefers dark mode');
    await userEvent.type(screen.getByPlaceholderText(/add a memory/i), 'New fact');
    await userEvent.click(screen.getByRole('button', { name: /add/i }));
    await waitFor(() => expect(api.addMemory).toHaveBeenCalledWith('New fact'));
    expect(await screen.findByText('New fact')).toBeInTheDocument();
  });

  it('deletes a memory', async () => {
    vi.mocked(api.deleteMemory).mockResolvedValue(undefined);
    render(<MemorySettings />);
    await screen.findByText('Prefers dark mode');
    await userEvent.click(screen.getByRole('button', { name: /delete/i }));
    await waitFor(() => expect(api.deleteMemory).toHaveBeenCalledWith('m1'));
    expect(screen.queryByText('Prefers dark mode')).not.toBeInTheDocument();
  });
});
```

**Note:** match the exact import alias your project uses for `@/lib/api` (check an existing test, e.g. `store.auth.test.ts`) and the existing testing-library setup; if `userEvent` is not configured in the project, use `fireEvent` from `@testing-library/react` instead.

- [ ] **Step 2: Run test to verify it fails**

Run (in `nextjs-frontend/`): `npx vitest run src/components/memory-settings.test.tsx`
Expected: FAIL — `MemorySettings` is not exported

- [ ] **Step 3: Implement types, API client, and component**

Append to `nextjs-frontend/src/lib/types.ts`:

```ts
export interface MemoryRecord {
  id: string;
  content: string;
  source: 'auto' | 'manual';
  created_at: string;
  updated_at: string;
}
```

Append to `nextjs-frontend/src/lib/api.ts`:

```ts
// ── User memory (personalization) ──────────────────────────────────────────

export async function getMemories(): Promise<MemoryRecord[]> {
  const res = await fetch(`${API_BASE}/api/memory`, { credentials: 'include', cache: 'no-store' });
  if (!res.ok) throw new Error(`Failed to load memories: ${res.status}`);
  return res.json();
}

export async function addMemory(content: string): Promise<MemoryRecord> {
  const res = await fetch(`${API_BASE}/api/memory`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify({ content }),
  });
  if (!res.ok) throw new Error(`Failed to add memory: ${res.status}`);
  return res.json();
}

export async function updateMemory(id: string, content: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/memory/${id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify({ content }),
  });
  if (!res.ok) throw new Error(`Failed to update memory: ${res.status}`);
}

export async function deleteMemory(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/memory/${id}`, {
    method: 'DELETE',
    credentials: 'include',
  });
  if (!res.ok) throw new Error(`Failed to delete memory: ${res.status}`);
}

export async function clearMemories(): Promise<{ deleted: number }> {
  const res = await fetch(`${API_BASE}/api/memory/clear`, {
    method: 'POST',
    credentials: 'include',
  });
  if (!res.ok) throw new Error(`Failed to clear memories: ${res.status}`);
  return res.json();
}
```

(Add `MemoryRecord` to the existing type imports at the top of `api.ts`.)

In `nextjs-frontend/src/components/settings-panel.tsx`, add a `MemorySettings` component and render it as a new `<Section title="Memory">` after the existing "Knowledge Base" section (line ~492):

```tsx
function MemorySettings() {
  const [memories, setMemories] = useState<MemoryRecord[]>([]);
  const [draft, setDraft] = useState('');
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setMemories(await getMemories());
      setError(null);
    } catch {
      setError('Could not load memories');
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const onAdd = async () => {
    const content = draft.trim();
    if (!content) return;
    try {
      await addMemory(content);
      setDraft('');
      await refresh();
    } catch {
      setError('Could not add memory');
    }
  };

  const onDelete = async (id: string) => {
    try {
      await deleteMemory(id);
      await refresh();
    } catch {
      setError('Could not delete memory');
    }
  };

  const onClearAll = async () => {
    if (!confirm('Delete all saved memories?')) return;
    try {
      await clearMemories();
      await refresh();
    } catch {
      setError('Could not clear memories');
    }
  };

  const onEdit = async (id: string, content: string) => {
    const next = prompt('Edit memory', content);
    if (next === null || !next.trim() || next === content) return;
    try {
      await updateMemory(id, next.trim());
      await refresh();
    } catch {
      setError('Could not update memory');
    }
  };

  return (
    <div className="space-y-2">
      {error && <p className="text-xs text-destructive">{error}</p>}
      {memories.length === 0 ? (
        <p className="text-xs text-muted-foreground">
          Nothing remembered yet. Facts you share in chat are saved here automatically.
        </p>
      ) : (
        <ul className="space-y-1">
          {memories.map((m) => (
            <li key={m.id} className="flex items-start justify-between gap-2 rounded border px-2 py-1.5 text-sm">
              <span>{m.content}</span>
              <span className="flex shrink-0 gap-1">
                <Button variant="ghost" size="sm" onClick={() => onEdit(m.id, m.content)}>
                  Edit
                </Button>
                <Button variant="ghost" size="sm" onClick={() => onDelete(m.id)}>
                  Delete
                </Button>
              </span>
            </li>
          ))}
        </ul>
      )}
      <div className="flex gap-2">
        <Input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Add a memory…"
          onKeyDown={(e) => {
            if (e.key === 'Enter') void onAdd();
          }}
        />
        <Button size="sm" onClick={() => void onAdd()}>
          Add
        </Button>
      </div>
      {memories.length > 0 && (
        <Button variant="outline" size="sm" onClick={() => void onClearAll()}>
          Clear all
        </Button>
      )}
    </div>
  );
}
```

Match the existing imports in `settings-panel.tsx` (`useState`, `useEffect`, `useCallback`, `Input`, `Button` are likely already imported — extend, don't duplicate) and export `MemorySettings` so the test can import it:

```tsx
export function MemorySettings() { ... }
```

Render inside the panel:

```tsx
<Section title="Memory">
  <MemorySettings />
</Section>
```

- [ ] **Step 4: Run test to verify it passes**

Run (in `nextjs-frontend/`): `npx vitest run src/components/memory-settings.test.tsx`
Expected: PASS (3 tests)

- [ ] **Step 5: Lint and commit**

```bash
cd nextjs-frontend && npm run lint && cd ..
git add nextjs-frontend/src/lib/types.ts nextjs-frontend/src/lib/api.ts nextjs-frontend/src/components/settings-panel.tsx nextjs-frontend/src/components/memory-settings.test.tsx
git commit -m "feat: memory management section in settings panel"
```

---

### Task 9: Documentation updates

**Files:**
- Modify: `CLAUDE.md`, `AGENTS.md`, `README.md`, `nextjs-frontend/CLAUDE.md`, `nextjs-frontend/AGENTS.md`

**Interfaces:**
- Consumes: final names/signatures from Tasks 1–8.
- Produces: docs that describe the shipped mechanics.

- [ ] **Step 1: Update root `CLAUDE.md`**

1. In **Directory layout**, add under `app/backend/` and `app/services/` entries:

```
    user_memory_store.py  Postgres user-memory persistence (user_memories table)
```

```
    user_memory_service.py  Persistent user memory: prompt block, LLM extraction, orchestration
```

and under `app/backend/api/`:

```
memory.py      /api/memory CRUD (user-level persistent memory)
```

2. Fix the stale persistence claim: wherever SQLite/`conversation_db.py` is described as the conversation store, state that the canonical store is **PostgreSQL** (`app/backend/conversation_store.py`, `Config.DATABASE_URL`) and that `conversation_db.py` is legacy.
3. Add an architecture subsection after "Conversation Memory":

```markdown
### Persistent user memory (`app/backend/user_memory_store.py`, `app/services/user_memory_service.py`)
Per-user durable memory in Postgres (`user_memories`, FK to `users` with ON DELETE CASCADE).
After each chat/voice exchange a fire-and-forget task asks the LLM gateway (request's own
backend/model) for JSON ops (add/update/delete, ≤3/turn) and applies them; extraction failure
never affects the response. On every chat/RAG/voice request the backend loads the user's
memories and prepends an "About the user" system block (cap 50). Explicit "remember that…"
requests are handled by the same extraction prompt. Manual management: `/api/memory` CRUD +
Settings → Memory panel. Distinct from session-scoped `conversation_memory.py`.
```

4. In the CI/CD or endpoints area, no change needed beyond the layout entry.

- [ ] **Step 2: Update root `AGENTS.md`**

Add one rule near any existing persistence/auth guidance:

```markdown
- Any new persisted data must be scoped by `user_id` (see `conversation_store.py` and
  `user_memory_store.py`); never add user-keyed tables without the FK + user filter.
```

- [ ] **Step 3: Update `README.md`**

Add a feature bullet in the features list:

```markdown
- **Persistent personalization** — LiteMindUI remembers durable facts you share (preferences, projects, corrections) across sessions and providers, injects them into every new chat/RAG/voice conversation, and lets you manage them in Settings → Memory.
```

- [ ] **Step 4: Update `nextjs-frontend/CLAUDE.md` and `nextjs-frontend/AGENTS.md`**

Add a short note (where components/lib are documented):

```markdown
- User memory management lives in `components/settings-panel.tsx` (`MemorySettings`) and
  `lib/api.ts` (`getMemories`/`addMemory`/`updateMemory`/`deleteMemory`/`clearMemories`),
  backed by `/api/memory` (cookie-authenticated, like the rest of the API client).
```

- [ ] **Step 5: Full verification + commit**

```bash
uv run pytest -x -q
uv run ruff check .
uv run ty check app/backend app/services app/core app/ingestion app/skills main.py config.py logging_config.py
cd nextjs-frontend && npm run lint && npm run build && cd ..
git add CLAUDE.md AGENTS.md README.md nextjs-frontend/CLAUDE.md nextjs-frontend/AGENTS.md
git commit -m "docs: document persistent user memory mechanics"
```

Expected: all backend tests pass (existing suite + new), lint/type-check clean, frontend lint + build succeed.
