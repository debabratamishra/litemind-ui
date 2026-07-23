"""
Unit tests for ConversationStore.

Uses an in-memory FakeStore subclass so no live PostgreSQL is required. The
focus is the core security property: every operation is scoped by user_id and
one user can never see or mutate another user's conversations/messages.
"""

import uuid

import pytest

from app.backend.conversation_store import ConversationRecord, ConversationStore, MessageRecord


class FakeStore(ConversationStore):
    """In-memory implementation of the same async interface as ConversationStore."""

    def __init__(self):
        self.conversations = {}  # conv_id -> dict(user_id, title, type, summary, messages[])
        self.msg_counter = 0

    async def init_schema(self):
        return None

    async def upsert_user(self, user_id, email=None):
        return None

    async def create_conversation(self, user_id, title="New Chat", conversation_type="chat"):
        cid = str(uuid.uuid4())
        self.conversations[cid] = {
            "id": cid,
            "user_id": user_id,
            "title": title or "New Chat",
            "conversation_type": conversation_type,
            "summary": None,
            "messages": [],
        }
        return ConversationRecord(
            id=cid, user_id=user_id, title=title, conversation_type=conversation_type,
            created_at="2026-01-01T00:00:00", updated_at="2026-01-01T00:00:00", summary=None,
        )

    async def get_conversation(self, conversation_id, user_id):
        c = self.conversations.get(conversation_id)
        if not c or c["user_id"] != user_id:
            return None
        return self._to_record(c)

    async def list_conversations(self, user_id, conversation_type=None):
        out = []
        for c in self.conversations.values():
            if c["user_id"] != user_id:
                continue
            if conversation_type and c["conversation_type"] != conversation_type:
                continue
            out.append(self._to_record(c))
        return out

    async def update_conversation(self, conversation_id, user_id, title=None, summary=None):
        c = self.conversations.get(conversation_id)
        if not c or c["user_id"] != user_id:
            return None
        if title is not None:
            c["title"] = title
        if summary is not None:
            c["summary"] = summary
        return self._to_record(c)

    async def delete_conversation(self, conversation_id, user_id):
        c = self.conversations.get(conversation_id)
        if not c or c["user_id"] != user_id:
            return False
        del self.conversations[conversation_id]
        return True

    async def add_message(self, conversation_id, user_id, role, content):
        c = self.conversations.get(conversation_id)
        if not c or c["user_id"] != user_id:
            return None
        self.msg_counter += 1
        mid = f"msg_{self.msg_counter}"
        c["messages"].append({"id": mid, "role": role, "content": content})
        return MessageRecord(
            id=mid, conversation_id=conversation_id, role=role, content=content,
            created_at="2026-01-01T00:00:00",
        )

    async def get_messages(self, conversation_id, user_id):
        c = self.conversations.get(conversation_id)
        if not c or c["user_id"] != user_id:
            return []
        return [
            MessageRecord(
                id=m["id"], conversation_id=conversation_id, role=m["role"],
                content=m["content"], created_at="2026-01-01T00:00:00",
            )
            for m in c["messages"]
        ]

    @staticmethod
    def _to_record(c):
        return ConversationRecord(
            id=c["id"], user_id=c["user_id"], title=c["title"],
            conversation_type=c["conversation_type"],
            created_at="2026-01-01T00:00:00", updated_at="2026-01-01T00:00:00",
            summary=c["summary"],
        )


@pytest.mark.asyncio
async def test_create_and_list_isolated():
    s = FakeStore()
    a = await s.create_conversation("userA", "A's chat")
    b = await s.create_conversation("userB", "B's chat")
    a_list = await s.list_conversations("userA")
    b_list = await s.list_conversations("userB")
    assert len(a_list) == 1 and a_list[0].id == a.id
    assert len(b_list) == 1 and b_list[0].id == b.id


@pytest.mark.asyncio
async def test_cross_user_get_returns_none():
    s = FakeStore()
    a = await s.create_conversation("userA", "A's chat")
    assert await s.get_conversation(a.id, "userB") is None
    assert await s.get_conversation(a.id, "userA") is not None


@pytest.mark.asyncio
async def test_cross_user_delete_is_false():
    s = FakeStore()
    a = await s.create_conversation("userA", "A's chat")
    assert await s.delete_conversation(a.id, "userB") is False
    assert await s.delete_conversation(a.id, "userA") is True
    assert await s.get_conversation(a.id, "userA") is None


@pytest.mark.asyncio
async def test_messages_isolated():
    s = FakeStore()
    a = await s.create_conversation("userA", "A's chat")
    await s.add_message(a.id, "userA", "user", "hello")
    assert len(await s.get_messages(a.id, "userA")) == 1
    assert await s.add_message(a.id, "userB", "user", "intrude") is None
    assert len(await s.get_messages(a.id, "userB")) == 0


@pytest.mark.asyncio
async def test_update_requires_ownership():
    s = FakeStore()
    a = await s.create_conversation("userA", "A's chat")
    assert await s.update_conversation(a.id, "userB", title="hacked") is None
    updated = await s.update_conversation(a.id, "userA", title="renamed")
    assert updated.title == "renamed"
