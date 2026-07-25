"""
Unit test for per-user session isolation in ConversationMemoryService.

Verifies that two different users with the SAME session_id get completely
separate in-memory contexts (the core multi-tenant isolation guarantee).
"""

import pytest

from app.services.conversation_memory import ConversationMemoryService


@pytest.fixture
def svc():
    return ConversationMemoryService()


@pytest.mark.asyncio
async def test_same_session_id_isolated_per_user(svc):
    a = svc.get_or_create_session("userA", "sess1")
    b = svc.get_or_create_session("userB", "sess1")
    assert a is not b
    assert a.session_id == "userA:sess1"
    assert b.session_id == "userB:sess1"

    svc.add_message("userA", "sess1", "user", "hello from A")
    svc.add_message("userB", "sess1", "user", "hello from B")

    a_msgs = svc.get_messages_for_context("userA", "sess1")
    b_msgs = svc.get_messages_for_context("userB", "sess1")
    assert len(a_msgs) == 1 and a_msgs[0]["content"] == "hello from A"
    assert len(b_msgs) == 1 and b_msgs[0]["content"] == "hello from B"


@pytest.mark.asyncio
async def test_clear_session_scoped_to_user(svc):
    svc.get_or_create_session("userA", "sess1")
    svc.get_or_create_session("userB", "sess1")
    # Clearing A's session must not touch B's.
    assert svc.clear_session("userA", "sess1") is True
    assert svc.clear_session("userB", "sess1") is True
    assert svc.clear_session("userA", "sess1") is False  # already gone


@pytest.mark.asyncio
async def test_stats_independent_per_user(svc):
    svc.add_message("userA", "sess1", "user", "A message one")
    svc.add_message("userA", "sess1", "user", "A message two")
    svc.add_message("userB", "sess1", "user", "B message one")

    a_stats = svc.get_session_stats("userA", "sess1")
    b_stats = svc.get_session_stats("userB", "sess1")
    assert a_stats["message_count"] == 2
    assert b_stats["message_count"] == 1
