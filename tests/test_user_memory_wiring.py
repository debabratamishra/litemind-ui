"""Chat wiring: memory block injection + post-exchange extraction trigger."""

from unittest.mock import AsyncMock, patch

import pytest

import app.backend.api.chat as chat_api
from app.backend.models.api_models import ChatRequestEnhanced


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
    # If there is a system message, it should not be the persistent memory block
    if sent_messages[0]["role"] == "system":
        assert "persistent memory" not in sent_messages[0]["content"]
    # Otherwise, the first message is not a system message (should be the user message)


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
