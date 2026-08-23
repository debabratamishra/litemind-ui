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
