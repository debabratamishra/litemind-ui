"""
User-scoped conversation persistence backed by PostgreSQL (asyncpg).

This replaces the previous SQLite ``conversation_db`` module. Every query is
filtered by ``user_id`` so that one user can never read or mutate another
user's conversations or messages — this is the core isolation guarantee for
the authentication feature.

The schema (users / conversations / messages) is created idempotently by
``init_schema()``. The source of truth for *users* is GoTrue; we mirror the
GoTrue ``sub`` (a UUID) into the local ``users`` table so conversations can
reference it with a foreign key.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import List, Optional

import asyncpg

from config import Config

logger = logging.getLogger(__name__)

# ── Schema ────────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY,
    email TEXT UNIQUE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL DEFAULT 'New Chat',
    conversation_type TEXT DEFAULT 'chat',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    summary TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_conversations_user_id
    ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id
    ON messages(conversation_id);
"""


# ── Record types ────────────────────────────────────────────────────────────────

@dataclass
class ConversationRecord:
    id: str
    user_id: str
    title: str
    conversation_type: str
    created_at: str
    updated_at: str
    summary: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "conversation_type": self.conversation_type,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "summary": self.summary,
        }


@dataclass
class MessageRecord:
    id: str
    conversation_id: str
    role: str
    content: str
    created_at: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
        }


# ── Store ───────────────────────────────────────────────────────────────────────

class ConversationStore:
    """Async PostgreSQL store for user-scoped conversations and messages."""

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
        """Create the users/conversations/messages tables if they don't exist."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(SCHEMA_SQL)
        logger.info("Conversation store schema initialized")

    async def upsert_user(self, user_id: str, email: Optional[str] = None) -> None:
        """Mirror a GoTrue user into the local users table (idempotent)."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO users (id, email, created_at)
                VALUES ($1, $2, NOW())
                ON CONFLICT (id) DO NOTHING
                """,
                uuid.UUID(user_id),
                email,
            )

    async def create_conversation(
        self,
        user_id: str,
        title: str = "New Chat",
        conversation_type: str = "chat",
    ) -> ConversationRecord:
        """Create a conversation owned by ``user_id`` (mirroring the user first)."""
        if not title:
            title = "New Chat"
        if conversation_type not in {"chat", "rag"}:
            conversation_type = "chat"

        await self.upsert_user(user_id)
        conv_id = uuid.uuid4()
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO conversations (id, user_id, title, conversation_type, created_at, updated_at)
                VALUES ($1, $2, $3, $4, NOW(), NOW())
                RETURNING id, user_id, title, conversation_type, created_at, updated_at, summary
                """,
                conv_id,
                uuid.UUID(user_id),
                title,
                conversation_type,
            )
        return self._row_to_conversation(row)

    async def get_conversation(self, conversation_id: str, user_id: str) -> Optional[ConversationRecord]:
        """Fetch a conversation only if it belongs to ``user_id``; else None."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, user_id, title, conversation_type, created_at, updated_at, summary
                FROM conversations
                WHERE id = $1 AND user_id = $2
                """,
                uuid.UUID(conversation_id),
                uuid.UUID(user_id),
            )
        return self._row_to_conversation(row) if row else None

    async def list_conversations(
        self, user_id: str, conversation_type: Optional[str] = None
    ) -> List[ConversationRecord]:
        """List a user's conversations (most recently updated first)."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            if conversation_type:
                rows = await conn.fetch(
                    """
                    SELECT id, user_id, title, conversation_type, created_at, updated_at, summary
                    FROM conversations
                    WHERE user_id = $1 AND conversation_type = $2
                    ORDER BY updated_at DESC
                    """,
                    uuid.UUID(user_id),
                    conversation_type,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT id, user_id, title, conversation_type, created_at, updated_at, summary
                    FROM conversations
                    WHERE user_id = $1
                    ORDER BY updated_at DESC
                    """,
                    uuid.UUID(user_id),
                )
        return [self._row_to_conversation(r) for r in rows]

    async def update_conversation(
        self,
        conversation_id: str,
        user_id: str,
        title: Optional[str] = None,
        summary: Optional[str] = None,
    ) -> Optional[ConversationRecord]:
        """Update title/summary for a conversation owned by ``user_id``."""
        sets: List[str] = []
        params: List = []
        if title is not None:
            sets.append("title = $3")
            params.append(title)
        if summary is not None:
            sets.append("summary = $3" if not sets else "summary = $4")
            params.append(summary)
        if not sets:
            return await self.get_conversation(conversation_id, user_id)

        params.insert(0, uuid.UUID(conversation_id))
        params.insert(1, uuid.UUID(user_id))
        sets.append("updated_at = NOW()")
        # Build the SET clause with safe positional params (title=$3, summary=$4).
        set_clause = ", ".join(sets)
        query = f"""
            UPDATE conversations
            SET {set_clause}
            WHERE id = $1 AND user_id = $2
            RETURNING id, user_id, title, conversation_type, created_at, updated_at, summary
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, *params)
        return self._row_to_conversation(row) if row else None

    async def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        """Delete a conversation owned by ``user_id``; returns True if it existed."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM conversations
                WHERE id = $1 AND user_id = $2
                """,
                uuid.UUID(conversation_id),
                uuid.UUID(user_id),
            )
        # asyncpg returns "DELETE <n>"
        return result.startswith("DELETE") and result.split()[1] not in ("0",)

    async def add_message(
        self, conversation_id: str, user_id: str, role: str, content: str
    ) -> Optional[MessageRecord]:
        """Append a message; verifies ownership of the conversation first."""
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"role must be one of user/assistant/system, got {role!r}")
        msg_id = uuid.uuid4()
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            # Ensure the conversation belongs to the user.
            owner = await conn.fetchval(
                "SELECT user_id FROM conversations WHERE id = $1",
                uuid.UUID(conversation_id),
            )
            if owner is None or str(owner) != str(uuid.UUID(user_id)):
                return None
            row = await conn.fetchrow(
                """
                INSERT INTO messages (id, conversation_id, role, content, created_at)
                VALUES ($1, $2, $3, $4, NOW())
                RETURNING id, conversation_id, role, content, created_at
                """,
                msg_id,
                uuid.UUID(conversation_id),
                role,
                content,
            )
            # Bump the conversation's updated_at for ordering.
            await conn.execute(
                "UPDATE conversations SET updated_at = NOW() WHERE id = $1",
                uuid.UUID(conversation_id),
            )
        return self._row_to_message(row)

    async def get_messages(self, conversation_id: str, user_id: str) -> List[MessageRecord]:
        """List messages for a conversation owned by ``user_id`` (oldest first)."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            owner = await conn.fetchval(
                "SELECT user_id FROM conversations WHERE id = $1",
                uuid.UUID(conversation_id),
            )
            if owner is None or str(owner) != str(uuid.UUID(user_id)):
                return []
            rows = await conn.fetch(
                """
                SELECT id, conversation_id, role, content, created_at
                FROM messages
                WHERE conversation_id = $1
                ORDER BY created_at ASC
                """,
                uuid.UUID(conversation_id),
            )
        return [self._row_to_message(r) for r in rows]

    # ── Helpers ──

    @staticmethod
    def _row_to_conversation(row) -> ConversationRecord:
        return ConversationRecord(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            title=row["title"],
            conversation_type=row["conversation_type"],
            created_at=row["created_at"].isoformat() if hasattr(row["created_at"], "isoformat") else str(row["created_at"]),
            updated_at=row["updated_at"].isoformat() if hasattr(row["updated_at"], "isoformat") else str(row["updated_at"]),
            summary=row["summary"],
        )

    @staticmethod
    def _row_to_message(row) -> MessageRecord:
        return MessageRecord(
            id=str(row["id"]),
            conversation_id=str(row["conversation_id"]),
            role=row["role"],
            content=row["content"],
            created_at=row["created_at"].isoformat() if hasattr(row["created_at"], "isoformat") else str(row["created_at"]),
        )


# ── Singleton ───────────────────────────────────────────────────────────────────

_store_instance: Optional[ConversationStore] = None


def get_conversation_store() -> ConversationStore:
    """Get the process-wide ConversationStore singleton."""
    global _store_instance
    if _store_instance is None:
        _store_instance = ConversationStore()
    return _store_instance
