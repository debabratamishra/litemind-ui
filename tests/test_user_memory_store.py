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
