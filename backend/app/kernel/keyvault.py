"""
Encrypted credential vault for user API keys (LLM/VLM/STT/TTS/embeddings/search).

Tokens are encrypted with Fernet (``cryptography``) before touching Postgres, so
a database dump alone never yields usable keys. The Fernet key comes from env
``KEYVAULT_KEY`` if set, else is generated once and persisted at
``<storage>/keyvault.key`` mode 0600 (storage path from ``backend.config.Config``).

Mirrors ``user_memory_store.py``: lazy connection-pool singleton, idempotent
schema init, module-level ``get_key_vault()`` accessor.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import List, Optional

import asyncpg
from cryptography.fernet import Fernet, InvalidToken

from backend.config import Config

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS user_credentials (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    kind TEXT NOT NULL,
    label TEXT NOT NULL DEFAULT '',
    provider TEXT NOT NULL DEFAULT '',
    token TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (kind, provider)
);
"""

_KEY_FILE_NAME = "keyvault.key"


def _resolve_fernet_key() -> bytes:
    """Env KEYVAULT_KEY wins; otherwise generate, persist 0600, reuse."""
    env_key = os.getenv("KEYVAULT_KEY")
    if env_key:
        return env_key.encode("utf-8")

    storage = Path(Config.get_storage_path())
    storage.mkdir(parents=True, exist_ok=True)
    key_path = storage / _KEY_FILE_NAME
    if key_path.exists():
        return key_path.read_bytes().strip()

    key = Fernet.generate_key()
    fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(key)
    finally:
        os.chmod(key_path, 0o600)
    return key


class KeyVault:
    """CRUD over ``user_credentials``; tokens stored only as Fernet ciphertext."""

    def __init__(self, dsn: Optional[str] = None):
        self._dsn = dsn or Config.DATABASE_URL
        self._pool: Optional[asyncpg.Pool] = None
        try:
            self._fernet = Fernet(_resolve_fernet_key())
        except Exception as exc:
            raise RuntimeError(
                "KEYVAULT_KEY is invalid: must be a urlsafe-base64-encoded 32-byte key"
            ) from exc

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

    async def set_credential(
        self, kind: str, provider: str, label: str, plaintext: str
    ) -> str:
        """Encrypt and store; upserts on (kind, provider). Returns row id."""
        token = self._fernet.encrypt(plaintext.encode("utf-8")).decode("ascii")
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO user_credentials (kind, label, provider, token)"
                " VALUES ($1, $2, $3, $4)"
                " ON CONFLICT (kind, provider) DO UPDATE SET"
                " label = EXCLUDED.label, token = EXCLUDED.token, updated_at = NOW()"
                " RETURNING id",
                kind,
                label,
                provider,
                token,
            )
        return str(row["id"])

    async def get_credential(self, kind: str, provider: str) -> Optional[str]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT token FROM user_credentials WHERE kind = $1 AND provider = $2"
                " ORDER BY updated_at DESC LIMIT 1",
                kind,
                provider,
            )
        if row is None:
            return None
        try:
            return self._fernet.decrypt(row["token"].encode("ascii")).decode("utf-8")
        except InvalidToken:
            logger.warning("corrupt credential token for kind=%s provider=%s", kind, provider)
            return None

    async def list_credentials(self) -> List[dict]:
        """Rows carry id/kind/label/provider/updated_at only — never the token."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, kind, label, provider, updated_at"
                " FROM user_credentials ORDER BY updated_at DESC"
            )
        return [
            {
                "id": str(row["id"]),
                "kind": row["kind"],
                "label": row["label"],
                "provider": row["provider"],
                "updated_at": str(row["updated_at"]),
            }
            for row in rows
        ]

    async def delete_credential(self, credential_id: str) -> bool:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "DELETE FROM user_credentials WHERE id = $1 RETURNING id",
                uuid.UUID(credential_id),
            )
        return row is not None


_vault_instance: Optional[KeyVault] = None


def get_key_vault() -> KeyVault:
    """Get the process-wide KeyVault singleton."""
    global _vault_instance
    if _vault_instance is None:
        _vault_instance = KeyVault()
    return _vault_instance
