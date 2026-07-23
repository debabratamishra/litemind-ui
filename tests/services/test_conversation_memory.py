"""
Unit tests for ``app.services.conversation_memory``.

Covers multi-turn context assembly, token estimation, the summarization
trigger (when token count exceeds 75% of the 24K limit), truncation, and
empty-history behaviour.

The summarization path calls ``stream_completion`` (an LLM boundary). That
boundary is mocked so no real network/LLM call is made.

All session-scoped methods now take a ``user_id`` as their first argument for
per-user isolation; these tests pin a single ``UID``.
"""

from app.services import conversation_memory as cm

UID = "test-user"


# ── ID / factories ───────────────────────────────────────────────────────────
def test_generate_session_id_unique_and_prefixed():
    a = cm.generate_session_id()
    b = cm.generate_session_id()
    assert a != b
    assert a.startswith("chat_")

    rag_id = cm.generate_session_id(prefix="rag")
    assert rag_id.startswith("rag_")


def test_message_and_context_dataclasses():
    msg = cm.Message(role="user", content="hi")
    assert msg.role == "user"
    assert msg.content == "hi"
    assert msg.token_count == 0
    assert msg.is_summary is False

    ctx = cm.ConversationContext(session_id="s", messages=[msg])
    assert len(ctx.messages) == 1
    assert ctx.session_id == "s"


def test_message_roundtrip_dict():
    msg = cm.Message(role="assistant", content="hello", token_count=5, is_summary=True)
    restored = cm.Message.from_dict(msg.to_dict())
    assert restored.role == "assistant"
    assert restored.content == "hello"
    assert restored.token_count == 5
    assert restored.is_summary is True


# ── Token estimation ────────────────────────────────────────────────────────
def test_estimate_tokens():
    svc = cm.ConversationMemoryService()
    assert svc.estimate_tokens("") == 0
    # ~4 chars per token, minimum 1 for non-empty text.
    assert svc.estimate_tokens("abcd") == 1
    assert svc.estimate_tokens("abcdefgh") == 2
    assert svc.estimate_tokens("x" * 8000) == 2000


# ── Session lifecycle ───────────────────────────────────────────────────────
def test_get_or_create_session_caches():
    svc = cm.ConversationMemoryService()
    s1 = svc.get_or_create_session(UID, "sess")
    s2 = svc.get_or_create_session(UID, "sess")
    assert s1 is s2
    assert s1.session_id == f"{UID}:sess"


def test_add_message_updates_tokens():
    svc = cm.ConversationMemoryService()
    msg = svc.add_message(UID, "sess", "user", "x" * 8000)  # 2000 tokens
    assert msg.token_count == 2000
    session = svc.get_or_create_session(UID, "sess")
    assert session.total_tokens == 2000
    assert len(session.messages) == 1


# ── Context assembly ────────────────────────────────────────────────────────
def test_get_messages_for_context_no_summary():
    svc = cm.ConversationMemoryService()
    svc.add_message(UID, "sess", "user", "question")
    svc.add_message(UID, "sess", "assistant", "answer")

    ctx = svc.get_messages_for_context(UID, "sess")
    assert [m["role"] for m in ctx] == ["user", "assistant"]
    assert ctx[0]["content"] == "question"
    assert ctx[1]["content"] == "answer"


def test_get_messages_for_context_includes_summary():
    svc = cm.ConversationMemoryService()
    svc.add_message(UID, "sess", "user", "question")
    session = svc.get_or_create_session(UID, "sess")
    session.summary = "Prior context summary"
    session.summary_tokens = 10

    ctx = svc.get_messages_for_context(UID, "sess")
    assert ctx[0]["role"] == "system"
    assert "Prior context summary" in ctx[0]["content"]
    assert len(ctx) == 2  # summary + 1 user message


def test_get_messages_for_context_excludes_summary_markers():
    svc = cm.ConversationMemoryService()
    svc.add_message(UID, "sess", "user", "real")
    svc.add_message(UID, "sess", "assistant", "summary text", is_summary=True)

    ctx = svc.get_messages_for_context(UID, "sess")
    contents = [m["content"] for m in ctx]
    assert "real" in contents
    assert "summary text" not in contents


def test_get_context_token_count():
    svc = cm.ConversationMemoryService()
    svc.add_message(UID, "sess", "user", "x" * 4000)  # 1000 tokens
    session = svc.get_or_create_session(UID, "sess")
    session.summary_tokens = 10

    assert svc.get_context_token_count(UID, "sess") == 1010


def test_prepare_context_for_llm_assembles_full_context():
    svc = cm.ConversationMemoryService()
    svc.add_message(UID, "sess", "user", "history line")
    session = svc.get_or_create_session(UID, "sess")
    session.summary = "old context"

    messages, total = svc.prepare_context_for_llm(
        UID, "sess", "new user message", system_prompt="You are helpful"
    )

    roles = [m["role"] for m in messages]
    assert roles[0] == "system"  # system prompt
    assert "You are helpful" in messages[0]["content"]
    assert roles[1] == "system"  # embedded summary
    assert "old context" in messages[1]["content"]
    assert {"role": "user", "content": "history line"} in [
        {"role": m["role"], "content": m["content"]} for m in messages
    ]
    assert messages[-1] == {"role": "user", "content": "new user message"}
    # total = system + summary + history + new message
    assert total > 0


def test_prepare_context_for_llm_empty_history():
    svc = cm.ConversationMemoryService()
    messages, total = svc.prepare_context_for_llm(UID, "fresh", "hello")
    # No system prompt, no summary -> only the new user message.
    assert messages == [{"role": "user", "content": "hello"}]
    assert total == svc.estimate_tokens("hello")


# ── Summarization trigger ───────────────────────────────────────────────────
def test_needs_summarization_below_threshold():
    svc = cm.ConversationMemoryService(max_context_tokens=24000, summarize_threshold=0.75)
    svc.add_message(UID, "sess", "user", "x" * 100)  # 25 tokens
    assert svc.needs_summarization(UID, "sess") is False


def test_needs_summarization_above_threshold():
    svc = cm.ConversationMemoryService(max_context_tokens=24000, summarize_threshold=0.75)
    # Threshold = 18000 tokens. 80000 chars -> 20000 tokens (>= 18000).
    svc.add_message(UID, "sess", "user", "x" * 80000)
    assert svc.needs_summarization(UID, "sess") is True


def test_needs_summarization_exactly_at_threshold():
    # 18000 tokens exactly -> int(24000*0.75) = 18000 -> >= threshold True.
    svc = cm.ConversationMemoryService(max_context_tokens=24000, summarize_threshold=0.75)
    svc.add_message(UID, "sess", "user", "x" * 72000)
    assert svc.needs_summarization(UID, "sess") is True


def test_default_constants():
    assert cm.DEFAULT_MAX_CONTEXT_TOKENS == 24000
    assert cm.DEFAULT_SUMMARIZE_THRESHOLD == 0.75


# ── Async summarization (LLM boundary mocked) ───────────────────────────────
async def test_summarize_if_needed_triggers_and_prunes(monkeypatch):
    svc = cm.ConversationMemoryService(max_context_tokens=24000, summarize_threshold=0.75)

    async def fake_stream(*args, **kwargs):
        yield "Summary of the conversation."

    monkeypatch.setattr(cm, "stream_completion", fake_stream)

    # 6 large messages => well over threshold, plenty to summarize.
    for _ in range(6):
        svc.add_message(UID, "sess", "user", "x" * 40000)  # 10000 tokens each

    assert svc.needs_summarization(UID, "sess") is True

    result = await svc.summarize_if_needed(UID, "sess")
    assert result is True

    session = svc.get_or_create_session(UID, "sess")
    assert session.summary == "Summary of the conversation."
    # keep_recent = 4 -> the 2 oldest are summarized away.
    assert len(session.messages) == 4
    assert session.total_tokens == 4 * 10000
    # Summary tokens counted in context total.
    assert svc.get_context_token_count(UID, "sess") == 40000 + session.summary_tokens


async def test_summarize_if_needed_noop_when_below_threshold(monkeypatch):
    svc = cm.ConversationMemoryService(max_context_tokens=24000, summarize_threshold=0.75)

    async def fake_stream(*args, **kwargs):
        yield "should not be called"

    monkeypatch.setattr(cm, "stream_completion", fake_stream)

    svc.add_message(UID, "sess", "user", "x" * 100)  # far below threshold

    result = await svc.summarize_if_needed(UID, "sess")
    assert result is False
    session = svc.get_or_create_session(UID, "sess")
    assert session.summary is None
    assert len(session.messages) == 1


async def test_summarize_if_needed_too_few_messages(monkeypatch):
    svc = cm.ConversationMemoryService(max_context_tokens=24000, summarize_threshold=0.75)

    async def fake_stream(*args, **kwargs):
        yield "should not be called"

    monkeypatch.setattr(cm, "stream_completion", fake_stream)

    # Over threshold but fewer than 4 messages -> nothing to summarize.
    for _ in range(3):
        svc.add_message(UID, "sess", "user", "x" * 40000)  # 10000 tokens each

    result = await svc.summarize_if_needed(UID, "sess")
    assert result is False


async def test_summarize_if_needed_force(monkeypatch):
    svc = cm.ConversationMemoryService(max_context_tokens=24000, summarize_threshold=0.75)

    async def fake_stream(*args, **kwargs):
        yield "Forced summary."

    monkeypatch.setattr(cm, "stream_completion", fake_stream)

    # 6 messages over threshold; force should summarize regardless.
    for _ in range(6):
        svc.add_message(UID, "sess", "user", "x" * 40000)

    result = await svc.summarize_if_needed(UID, "sess", force=True)
    assert result is True
    assert svc.get_or_create_session(UID, "sess").summary == "Forced summary."


async def test_summarize_if_needed_handles_llm_failure(monkeypatch):
    svc = cm.ConversationMemoryService(max_context_tokens=24000, summarize_threshold=0.75)

    async def failing_stream(*args, **kwargs):
        raise RuntimeError("LLM down")
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(cm, "stream_completion", failing_stream)

    for _ in range(6):
        svc.add_message(UID, "sess", "user", "x" * 40000)

    # LLM failure -> no summary, but does not raise; returns False.
    result = await svc.summarize_if_needed(UID, "sess")
    assert result is False
    assert svc.get_or_create_session(UID, "sess").summary is None


# ── Session stats / clear ───────────────────────────────────────────────────
def test_get_session_stats_and_clear():
    svc = cm.ConversationMemoryService()
    svc.add_message(UID, "sess", "user", "x" * 4000)  # 1000 tokens
    stats = svc.get_session_stats(UID, "sess")
    assert stats["message_count"] == 1
    assert stats["total_tokens"] == 1000
    assert stats["has_summary"] is False

    assert svc.clear_session(UID, "sess") is True
    # Session removed.
    assert svc.clear_session(UID, "sess") is False


# ── Singleton ───────────────────────────────────────────────────────────────
def test_get_memory_service_returns_service(monkeypatch):
    monkeypatch.setattr(cm, "_memory_service", None)
    svc = cm.get_memory_service()
    assert isinstance(svc, cm.ConversationMemoryService)
    # Cached on repeat call.
    assert cm.get_memory_service() is svc
    monkeypatch.setattr(cm, "_memory_service", None)
