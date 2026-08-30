"""Unit tests for UserMemoryStore.

Uses an in-memory FakeStore subclass (same approach as
tests/test_conversation_store.py) so no live PostgreSQL is required. The focus
is user isolation: one user can never read or mutate another user's memories.
"""

import os
import uuid

import pytest

from app.backend.user_memory_store import UserMemoryRecord, UserMemoryStore


class FakeUserMemoryStore(UserMemoryStore):
    """In-memory implementation of the same async interface."""

    def __init__(self):
        self.memories = {}  # id -> dict(row fields)
        self._next_timestamp = 0

    async def init_schema(self):
        return None

    async def list_memories(self, user_id, limit=50):
        rows = [m for m in self.memories.values() if m["user_id"] == user_id]
        rows.sort(key=lambda m: m["created_at"], reverse=True)
        return [self._record(m) for m in rows[:limit]]

    async def add_memory(self, user_id, content, source="auto"):
        mid = str(uuid.uuid4())
        timestamp = f"2026-08-23T00:00:{self._next_timestamp:02d}"
        row = {
            "id": mid, "user_id": user_id, "content": content, "source": source,
            "created_at": timestamp, "updated_at": timestamp,
        }
        self.memories[mid] = row
        self._next_timestamp += 1
        return self._record(row)

    async def update_memory(self, user_id, memory_id, content):
        m = self.memories.get(memory_id)
        if not m or m["user_id"] != user_id:
            return False
        m["content"] = content
        timestamp = f"2026-08-23T00:00:{self._next_timestamp:02d}"
        m["updated_at"] = timestamp
        self._next_timestamp += 1
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


# ── Real-store integration (spec §8) ─────────────────────────────────────────

_REAL_DSN = os.environ.get("TEST_DATABASE_URL", "")
_requires_postgres = pytest.mark.skipif(
    not _REAL_DSN, reason="set TEST_DATABASE_URL to run real-store integration tests"
)


@_requires_postgres
@pytest.mark.asyncio
async def test_real_store_crud_and_cross_user_isolation():
    """Runs only when TEST_DATABASE_URL points at a disposable Postgres."""
    import asyncpg

    from app.backend.conversation_store import SCHEMA_SQL as CORE_SCHEMA_SQL
    from app.backend.user_memory_store import UserMemoryStore

    store = UserMemoryStore(_REAL_DSN)
    # user_memories has an FK on users(id); make sure both tables exist and
    # create real user rows so the inserts below satisfy the constraint.
    pool = await asyncpg.create_pool(_REAL_DSN)
    try:
        async with pool.acquire() as conn:
            await conn.execute(CORE_SCHEMA_SQL)
            await store.init_schema()
            user_a, user_b = str(uuid.uuid4()), str(uuid.uuid4())
            await conn.executemany(
                "INSERT INTO users (id, email) VALUES ($1, $2)",
                [(uuid.UUID(user_a), f"{user_a}@test.local"), (uuid.UUID(user_b), f"{user_b}@test.local")],
            )
        try:
            rec = await store.add_memory(user_a, "likes tea")
            assert rec.source == "auto"
            assert [r.content for r in await store.list_memories(user_a)] == ["likes tea"]
            # B sees nothing of A's, and cannot mutate or delete it
            assert await store.list_memories(user_b) == []
            assert not await store.update_memory(user_b, rec.id, "hacked")
            assert not await store.delete_memory(user_b, rec.id)
            assert (await store.list_memories(user_a))[0].content == "likes tea"
        finally:
            await store.clear_memories(user_a)
            await store.clear_memories(user_b)
            async with pool.acquire() as conn:
                await conn.execute("DELETE FROM users WHERE id = ANY($1::uuid[])", [user_a, user_b])
    finally:
        await pool.close()
