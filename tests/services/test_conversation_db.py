"""
Unit tests for ``app.services.conversation_db``.

Covers SQLite CRUD for conversations and messages, message ordering,
persistence across reopening the database file, and corrupt-DB handling.

All tests are fully offline: they use a real temporary SQLite file (that is
the unit under test) with no network or external services involved.
"""
import sqlite3
import time
from typing import Any

import pytest

from app.services import conversation_db as cdb


# ── ID generation ──────────────────────────────────────────────────────────
def test_generate_conversation_id_unique():
    a = cdb.generate_conversation_id()
    b = cdb.generate_conversation_id()
    assert isinstance(a, str)
    assert a != b
    assert a.startswith("conv_")
    assert b.startswith("conv_")


def test_generate_message_id_unique():
    a = cdb.generate_message_id()
    b = cdb.generate_message_id()
    assert isinstance(a, str)
    assert a != b
    assert a.startswith("msg_")


# ── Schema / init ───────────────────────────────────────────────────────────
def test_init_creates_file_and_schema(tmp_path):
    db_path = tmp_path / "conv.db"
    assert not db_path.exists()
    db = cdb.ConversationDatabase(db_path)
    # File is created and tables exist.
    assert db_path.exists()
    with db._get_connection() as conn:
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert {"conversations", "messages"} <= tables


# ── Conversation create / get / list / update / delete ──────────────────────
def test_create_and_get_conversation(tmp_path):
    db = cdb.ConversationDatabase(tmp_path / "conv.db")
    conv = db.create_conversation("My Chat", conversation_type="chat")
    assert conv.id.startswith("conv_")
    assert conv.title == "My Chat"
    assert conv.conversation_type == "chat"

    fetched = db.get_conversation(conv.id)
    assert fetched is not None
    assert fetched.id == conv.id
    assert fetched.title == "My Chat"
    assert fetched.messages == []


def test_create_conversation_validation(tmp_path):
    db = cdb.ConversationDatabase(tmp_path / "conv.db")

    with pytest.raises(ValueError):
        db.create_conversation("")  # empty title

    bad_title: Any = 123
    with pytest.raises(ValueError):
        db.create_conversation(bad_title)  # non-string title

    with pytest.raises(ValueError):
        db.create_conversation("x" * 501)  # too long

    with pytest.raises(ValueError):
        db.create_conversation("ok", conversation_type="bogus")


def test_get_conversation_missing_returns_none(tmp_path):
    db = cdb.ConversationDatabase(tmp_path / "conv.db")
    assert db.get_conversation("does_not_exist") is None


def test_list_and_count_conversations(tmp_path):
    db = cdb.ConversationDatabase(tmp_path / "conv.db")
    db.create_conversation("Chat A", conversation_type="chat")
    db.create_conversation("Chat B", conversation_type="rag")
    db.create_conversation("Chat C", conversation_type="chat")

    all_conv = db.list_conversations()
    assert len(all_conv) == 3

    chats = db.list_conversations(conversation_type="chat")
    assert len(chats) == 2

    assert db.get_conversation_count() == 3
    assert db.get_conversation_count(conversation_type="rag") == 1


def test_list_conversations_pagination(tmp_path):
    db = cdb.ConversationDatabase(tmp_path / "conv.db")
    for i in range(5):
        db.create_conversation(f"Chat {i}")

    page = db.list_conversations(limit=2, offset=1)
    assert len(page) == 2


def test_update_conversation(tmp_path):
    db = cdb.ConversationDatabase(tmp_path / "conv.db")
    conv = db.create_conversation("Original Title")

    assert db.update_conversation(conv.id, title="New Title") is True
    assert db.update_conversation(conv.id, summary="a summary") is True
    assert db.update_conversation(conv.id, metadata={"k": "v"}) is True

    fetched = db.get_conversation(conv.id)
    assert fetched is not None
    assert fetched.title == "New Title"
    assert fetched.summary == "a summary"
    assert fetched.metadata == {"k": "v"}

    # No-op update returns True without error.
    assert db.update_conversation(conv.id) is True

    # Update missing conversation returns False.
    assert db.update_conversation("missing", title="x") is False


def test_delete_conversation(tmp_path):
    db = cdb.ConversationDatabase(tmp_path / "conv.db")
    conv = db.create_conversation("To Delete")
    db.add_message(conv.id, "user", "hello")

    assert db.delete_conversation(conv.id) is True
    assert db.get_conversation(conv.id) is None
    # Messages were cascade-deleted.
    assert db.get_messages(conv.id) == []

    # Deleting again returns False.
    assert db.delete_conversation(conv.id) is False


# ── Message add / get / delete ──────────────────────────────────────────────
def test_add_and_get_messages(tmp_path):
    db = cdb.ConversationDatabase(tmp_path / "conv.db")
    conv = db.create_conversation("Chat")

    db.add_message(conv.id, "user", "hi")
    db.add_message(conv.id, "assistant", "hello there")

    conv = db.get_conversation(conv.id)
    assert conv is not None
    assert len(conv.messages) == 2
    assert conv.messages[0].role == "user"
    assert conv.messages[0].content == "hi"
    assert conv.messages[1].role == "assistant"
    assert conv.messages[1].content == "hello there"


def test_add_message_validation(tmp_path):
    db = cdb.ConversationDatabase(tmp_path / "conv.db")

    with pytest.raises(ValueError):
        db.add_message("", "user", "x")  # empty conversation id

    bad_id: Any = 123
    with pytest.raises(ValueError):
        db.add_message(bad_id, "user", "x")  # non-string conversation id

    with pytest.raises(ValueError):
        db.add_message("c", "bogus", "x")  # invalid role

    bad_content: Any = 123
    with pytest.raises(ValueError):
        db.add_message("c", "user", bad_content)  # non-string content

    with pytest.raises(ValueError):
        db.add_message("c", "user", "x" * 1_000_001)  # too long


def test_message_metadata_roundtrip(tmp_path):
    db = cdb.ConversationDatabase(tmp_path / "conv.db")
    conv = db.create_conversation("Chat")
    db.add_message(conv.id, "user", "hi", metadata={"source": "web"})

    msg = db.get_messages(conv.id)[0]
    assert msg.metadata == {"source": "web"}


def test_get_messages_ordering_preserved(tmp_path):
    db = cdb.ConversationDatabase(tmp_path / "conv.db")
    conv = db.create_conversation("Chat")

    # Insert distinct content with a delay so created_at timestamps differ
    # and ORDER BY created_at ASC is deterministic.
    contents = [f"message number {i}" for i in range(10)]
    for c in contents:
        db.add_message(conv.id, "user", c)
        time.sleep(0.005)

    fetched = [m.content for m in db.get_messages(conv.id)]
    assert fetched == contents


def test_get_messages_limit(tmp_path):
    db = cdb.ConversationDatabase(tmp_path / "conv.db")
    conv = db.create_conversation("Chat")
    for i in range(5):
        db.add_message(conv.id, "user", f"m{i}")
        time.sleep(0.005)

    limited = db.get_messages(conv.id, limit=3)
    assert len(limited) == 3


def test_delete_message(tmp_path):
    db = cdb.ConversationDatabase(tmp_path / "conv.db")
    conv = db.create_conversation("Chat")
    msg = db.add_message(conv.id, "user", "delete me")

    assert db.delete_message(msg.id) is True
    assert db.get_messages(conv.id) == []

    # Deleting again returns False.
    assert db.delete_message(msg.id) is False


# ── Persistence across reopen ──────────────────────────────────────────────
def test_persistence_across_reopen(tmp_path):
    db_path = tmp_path / "conv.db"
    db = cdb.ConversationDatabase(db_path)
    conv = db.create_conversation("Persistent Chat")
    db.add_message(conv.id, "user", "first")
    time.sleep(0.005)
    db.add_message(conv.id, "assistant", "second")
    del db  # drop the handle; file stays on disk

    reopened = cdb.ConversationDatabase(db_path)
    loaded = reopened.get_messages(conv.id)
    assert len(loaded) == 2
    assert loaded[0].content == "first"
    assert loaded[1].content == "second"


# ── Search ──────────────────────────────────────────────────────────────────
def test_search_conversations_by_title_and_content(tmp_path):
    db = cdb.ConversationDatabase(tmp_path / "conv.db")
    conv = db.create_conversation("Project Alpha")
    db.add_message(conv.id, "user", "tell me about the budget")
    time.sleep(0.005)
    other = db.create_conversation("Unrelated")
    db.add_message(other.id, "user", "random thought")

    by_title = db.search_conversations("Alpha")
    assert any(c.id == conv.id for c in by_title)

    by_content = db.search_conversations("budget")
    assert any(c.id == conv.id for c in by_content)

    by_type = db.search_conversations("thought", conversation_type="chat")
    assert any(c.id == other.id for c in by_type)


def test_search_escapes_like_wildcards(tmp_path):
    db = cdb.ConversationDatabase(tmp_path / "conv.db")
    # A title containing LIKE wildcard characters must not match everything.
    conv = db.create_conversation("100% done_here")
    results = db.search_conversations("%")
    assert any(c.id == conv.id for c in results)

    no_match = db.search_conversations("_xyz")
    assert all("%" not in c.title for c in no_match)


# ── Title generation ────────────────────────────────────────────────────────
def test_generate_title_from_message(tmp_path):
    db = cdb.ConversationDatabase(tmp_path / "conv.db")
    assert db.generate_title_from_message("Hello world this is long", max_words=3) == "Hello world this..."
    assert db.generate_title_from_message("   ") == "New Chat"
    # Very long single token is truncated.
    assert len(db.generate_title_from_message("x" * 100)) <= 40


# ── Corrupt DB handling ─────────────────────────────────────────────────────
def test_corrupt_db_raises_gracefully(tmp_path):
    """A file that is not a valid SQLite database must produce a clear error
    rather than silently succeeding or returning wrong data."""
    corrupt = tmp_path / "corrupt.db"
    corrupt.write_bytes(b"not a real sqlite database" * 64)

    with pytest.raises(sqlite3.Error):
        db = cdb.ConversationDatabase(corrupt)
        db.get_conversation_count()


# ── Singleton ────────────────────────────────────────────────────────────────
def test_get_conversation_db_singleton(tmp_path, monkeypatch):
    temp_db = tmp_path / "singleton.db"
    monkeypatch.setattr(cdb, "DEFAULT_DB_PATH", temp_db)
    monkeypatch.setattr(cdb, "_db_instance", None)

    first = cdb.get_conversation_db()
    second = cdb.get_conversation_db()
    assert isinstance(first, cdb.ConversationDatabase)
    assert first is second  # cached

    monkeypatch.setattr(cdb, "_db_instance", None)
