"""Unit tests for the kernel KeyVault (Fernet-encrypted credential store).

Offline: a FakeConn/FakePool pair stands in for asyncpg so no live PostgreSQL
is required. Focus: set/get roundtrip, list never exposes token/ciphertext,
delete works, unknown -> None, and the Fernet key file is persisted at
<storage>/keyvault.key mode 0600 and reused by later KeyVault instances.
"""

import os
import stat
import uuid

import pytest

from backend.app.kernel.keyvault import KeyVault, get_key_vault
from backend.config import Config

# ── Fakes (same pattern as tests/test_conversation_store.py) ─────────────────


class FakeConn:
    """Tiny in-memory user_credentials table."""

    def __init__(self, rows):
        self._rows = rows  # id -> dict

    async def execute(self, sql, *params):
        return "CREATE TABLE"

    async def fetchrow(self, sql, *params):
        if sql.startswith("SELECT token"):
            for row in self._rows.values():
                if row["kind"] == params[0] and row["provider"] == params[1]:
                    return row
            return None
        if sql.startswith("INSERT"):
            # INSERT ... ON CONFLICT (kind, provider) DO UPDATE
            kind, label, provider, token = params
            for row in self._rows.values():
                if row["kind"] == kind and row["provider"] == provider:
                    row["label"], row["token"] = label, token
                    row["updated_at"] = "2026-01-01T00:00:01+00:00"
                    return row
            row = {
                "id": str(uuid.uuid4()),
                "kind": kind,
                "label": label,
                "provider": provider,
                "token": token,
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
            self._rows[row["id"]] = row
            return row
        if sql.startswith("DELETE"):
            return self._rows.pop(str(params[0]), None)
        raise AssertionError(f"unexpected SQL: {sql}")

    async def fetch(self, sql, *params):
        return list(self._rows.values())


class FakeAcquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class FakePool:
    def __init__(self):
        self.conn = FakeConn({})

    def acquire(self):
        return FakeAcquire(self.conn)


# ── CRUD ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_set_get_roundtrip():
    pool = FakePool()
    vault = KeyVault()
    vault._pool = pool
    cid = await vault.set_credential("llm", "openrouter", "main", "sk-secret-1")
    assert await vault.get_credential("llm", "openrouter") == "sk-secret-1"
    assert cid


@pytest.mark.asyncio
async def test_set_updates_existing_kind_provider():
    pool = FakePool()
    vault = KeyVault()
    vault._pool = pool
    await vault.set_credential("llm", "openrouter", "main", "sk-1")
    await vault.set_credential("llm", "openrouter", "renamed", "sk-2")
    assert await vault.get_credential("llm", "openrouter") == "sk-2"
    assert len(pool.conn._rows) == 1


@pytest.mark.asyncio
async def test_list_credentials_has_no_token_field():
    vault = KeyVault()
    vault._pool = FakePool()
    await vault.set_credential("tts", "kokoro", "voice", "plain-token")
    rows = await vault.list_credentials()
    assert len(rows) == 1
    assert set(rows[0]) == {"id", "kind", "label", "provider", "updated_at"}
    assert rows[0]["kind"] == "tts"
    assert rows[0]["provider"] == "kokoro"


@pytest.mark.asyncio
async def test_delete_removes_credential():
    pool = FakePool()
    vault = KeyVault()
    vault._pool = pool
    cid = await vault.set_credential("stt", "whisper", "main", "tok")
    assert await vault.delete_credential(cid) is True
    assert await vault.get_credential("stt", "whisper") is None
    assert await vault.delete_credential(cid) is False


@pytest.mark.asyncio
async def test_get_unknown_returns_none():
    vault = KeyVault()
    vault._pool = FakePool()
    assert await vault.get_credential("llm", "nope") is None


# ── Key file persistence ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_key_file_persisted_0600_and_reused_by_second_instance(tmp_path, monkeypatch):
    monkeypatch.delenv("KEYVAULT_KEY", raising=False)
    monkeypatch.setattr(Config, "get_storage_path", classmethod(lambda cls: str(tmp_path)))

    pool = FakePool()
    v1 = KeyVault()
    v1._pool = pool
    v2 = KeyVault()
    v2._pool = pool

    key_path = tmp_path / "keyvault.key"
    assert key_path.exists()
    assert stat.S_IMODE(os.stat(key_path).st_mode) == 0o600

    await v1.set_credential("embeddings", "openrouter", "embed", "emb-secret")
    # Second instance reads the same key file, so decryption still works.
    assert await v2.get_credential("embeddings", "openrouter") == "emb-secret"


def test_get_key_vault_singleton():
    assert get_key_vault() is get_key_vault()


def test_malformed_keyvault_key_raises_clear_error(monkeypatch):
    monkeypatch.setenv("KEYVAULT_KEY", "not-a-valid-fernet-key")
    with pytest.raises(RuntimeError, match="KEYVAULT_KEY is invalid"):
        KeyVault()
