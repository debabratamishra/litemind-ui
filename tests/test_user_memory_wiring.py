"""Chat wiring: memory block injection + post-exchange extraction trigger."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import app.backend.api.chat as chat_api
from app.backend.models.api_models import ChatRequestEnhanced, RAGQueryRequestEnhanced


async def aiter(items):
    """Wrap a plain list as an async iterator (shadows the builtin, which needs an async iterable)."""
    for item in items:
        yield item


async def _empty(*_args, **_kwargs):
    """Async generator yielding nothing (for mocking streaming calls)."""
    return
    yield


def _request(**overrides):
    base = dict(message="Remember I like tea", backend="ollama", model="gemma3:1b")
    base.update(overrides)
    return ChatRequestEnhanced(**base)


@pytest.mark.asyncio
async def test_stream_injects_memory_block():
    # Mock stream_completion to return an async generator that yields "Hello"
    # and capture the messages argument
    async def mock_stream(*args, **kwargs):
        mock_stream.messages = args[0]  # first argument is messages
        yield "Hello"

    mock_stream.messages = None

    with patch.object(chat_api, "load_memory_block", new=AsyncMock(return_value="About the user:\n- Likes tea")), \
         patch.object(chat_api, "stream_completion", mock_stream):
        chunks = [c async for c in chat_api._stream_chat_response(_request(), user_id="u-1")]
    assert chunks == ["Hello"]
    # Check that the first argument to stream_completion (the messages list) starts with the memory block
    sent_messages = mock_stream.messages
    assert sent_messages[0] == {"role": "system", "content": "About the user:\n- Likes tea"}


@pytest.mark.asyncio
async def test_stream_skips_block_when_empty():
    # Mock stream_completion to return an async generator that yields "Hi"
    async def mock_stream(*args, **kwargs):
        mock_stream.messages = args[0]  # first argument is messages
        yield "Hi"

    mock_stream.messages = None

    with patch.object(chat_api, "load_memory_block", new=AsyncMock(return_value="")), \
         patch.object(chat_api, "stream_completion", mock_stream):
        # We only need the first chunk to verify the call
        chunk = [c async for c in chat_api._stream_chat_response(_request(), user_id="u-1")][0]
    assert chunk == "Hi"
    # Check that the first argument to stream_completion does not contain a system block for persistent memory
    sent_messages = mock_stream.messages
    system_blocks = [m for m in sent_messages if m["role"] == "system"]
    assert all("persistent memory" not in m["content"] for m in system_blocks)


@pytest.mark.asyncio
async def test_non_stream_path_triggers_extraction():
    with patch.object(chat_api, "load_memory_block", new=AsyncMock(return_value="")), \
         patch.object(chat_api, "complete_text", new=AsyncMock(return_value="Enjoy your tea!")), \
         patch.object(chat_api.asyncio, "create_task") as mock_task, \
         patch.object(chat_api, "run_memory_update", new=AsyncMock()):
        await chat_api._handle_chat_request(_request(), user_id="u-1")
    assert mock_task.called


@pytest.mark.asyncio
async def test_handle_chat_request_without_user_skips_memory_load():
    with patch.object(chat_api, "load_memory_block", new=AsyncMock(return_value="")) as mock_load, \
         patch.object(chat_api, "complete_text", new=AsyncMock(return_value="ok")):
        await chat_api._handle_chat_request(_request())
    mock_load.assert_not_called()


# ── RAG injection ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rag_query_injects_memory_block():
    """query(..., memory_block=...) must place the block before conversation history."""
    from app.services import rag_service as rag_module
    from app.services.rag_service import RAGService

    svc = RAGService.__new__(RAGService)  # retrieval + prompt seams are patched below

    block = {"role": "system", "content": "About the user:\n- Likes tea"}
    with patch.object(svc, "get_retrieval_records", return_value=[]), \
         patch.object(svc, "build_grounded_user_prompt", return_value="grounded"), \
         patch.object(rag_module, "stream_completion", new=MagicMock(return_value=aiter(["answer"]))):
        chunks = [
            c
            async for c in svc.query(
                "what is x",
                system_prompt="You are helpful.",
                messages=[{"role": "user", "content": "hi"}],
                memory_block="About the user:\n- Likes tea",
            )
        ]
        sent = rag_module.stream_completion.call_args.args[0]

    assert "answer" in chunks
    assert block in sent
    assert sent.index(block) < sent.index({"role": "user", "content": "hi"})


@pytest.mark.asyncio
async def test_standard_rag_skill_forwards_memory_block_to_query():
    """StandardRAGSkill.stream must pass memory_block into rag_service.query."""
    from app.skills.rag import StandardRAGSkill

    mock_rag_service = MagicMock()
    mock_rag_service.query.side_effect = _empty

    request = RAGQueryRequestEnhanced(query="what is x")
    skill = StandardRAGSkill()
    chunks = [
        c
        async for c in skill.stream(request, mock_rag_service, memory_block="About the user:\n- Likes tea")
    ]

    assert chunks == []
    assert mock_rag_service.query.call_args.kwargs["memory_block"] == "About the user:\n- Likes tea"


# ── Voice pipeline ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_voice_pipeline_prepends_memory_to_system_instruction():
    from app.services.voice_pipeline import VoiceSettings

    settings = VoiceSettings(user_id="u-1", system_instruction="You are a helpful voice assistant.")
    with patch("app.services.voice_pipeline.load_memory_block", new=AsyncMock(return_value="About the user:\n- Likes tea")):
        from app.services.voice_pipeline import apply_memory_to_voice_settings

        await apply_memory_to_voice_settings(settings)
    assert settings.system_instruction.startswith("You are a helpful voice assistant.")
    assert "Likes tea" in settings.system_instruction


@pytest.mark.asyncio
async def test_voice_pipeline_skips_memory_without_user():
    from app.services.voice_pipeline import VoiceSettings, apply_memory_to_voice_settings

    settings = VoiceSettings(user_id=None, system_instruction="base")
    with patch("app.services.voice_pipeline.load_memory_block", new=AsyncMock(return_value="X")) as m:
        await apply_memory_to_voice_settings(settings)
    m.assert_not_called()
    assert settings.system_instruction == "base"
