"""Tests for user_memory_service pure functions."""

from unittest.mock import AsyncMock, patch

import pytest

import app.services.user_memory_service as ums
from app.services.user_memory_service import (
    MAX_MEMORY_OPS_PER_TURN,
    apply_memory_ops,
    build_memory_block,
    extract_memory_ops,
    load_memory_block,
    parse_memory_ops,
    run_memory_update,
)


class FakeMemory:
    def __init__(self, content):
        self.content = content


def test_build_memory_block_empty():
    assert build_memory_block([]) == ""


def test_build_memory_block_formats_facts():
    block = build_memory_block([FakeMemory("Prefers concise answers"), FakeMemory("Name is Alex")])
    assert block.startswith("Things you know about this user")
    assert "- Prefers concise answers" in block
    assert "- Name is Alex" in block


def test_parse_memory_ops_valid_json():
    raw = '[{"op": "add", "content": "Likes Rust"}]'
    assert parse_memory_ops(raw) == [{"op": "add", "content": "Likes Rust"}]


def test_parse_memory_ops_strips_code_fence():
    memory_id = "3fa85f64-5717-4562-b3fc-2c963f66afa6"
    raw = '```json\n[{"op": "delete", "id": "%s"}]\n```' % memory_id
    assert parse_memory_ops(raw) == [{"op": "delete", "id": memory_id}]


def test_parse_memory_ops_invalid_json_returns_empty():
    assert parse_memory_ops("not json at all") == []
    assert parse_memory_ops("") == []


def test_parse_memory_ops_rejects_bad_shapes():
    raw = """
    [
      {"op": "add"},
      {"op": "add", "content": ""},
      {"op": "bogus", "content": "x"},
      {"op": "update", "content": "no id"},
      {"op": "delete", "id": 42},
      {"op": "add", "content": 7},
      "just a string",
      {"op": "add", "content": "Valid one"}
    ]
    """
    assert parse_memory_ops(raw) == [{"op": "add", "content": "Valid one"}]


def test_parse_memory_ops_caps_at_three():
    raw = "[" + ",".join(f'{{"op": "add", "content": "c{i}"}}' for i in range(10)) + "]"
    assert len(parse_memory_ops(raw)) == MAX_MEMORY_OPS_PER_TURN


VALID_MEMORY_ID = "3fa85f64-5717-4562-b3fc-2c963f66afa6"


def test_parse_memory_ops_drops_mangled_uuid_but_keeps_siblings():
    """A mangled UUID on one op must not abort the whole batch (local models do this)."""
    raw = (
        '[{"op": "delete", "id": "not-a-uuid"},'
        f'{{"op": "add", "content": "Kept"}},'
        f'{{"op": "update", "id": "{VALID_MEMORY_ID}", "content": "Edited"}}]'
    )
    assert parse_memory_ops(raw) == [
        {"op": "add", "content": "Kept"},
        {"op": "update", "id": VALID_MEMORY_ID, "content": "Edited"},
    ]


def test_parse_memory_ops_truncates_overlong_content():
    raw = '[{"op": "add", "content": "' + "x" * 600 + '"}]'
    ops = parse_memory_ops(raw)
    assert len(ops[0]["content"]) == 500


# ── Extraction and orchestration (Task 3) ──────────────────────



class FakeStore:
    def __init__(self, memories=None):
        self.memories = list(memories or [])
        self.added = []
        self.updated = []
        self.deleted = []

    async def list_memories(self, user_id, limit=50):
        return self.memories

    async def add_memory(self, user_id, content, source="auto"):
        self.added.append((user_id, content, source))
        return type('R', (), {'content': content})()

    async def update_memory(self, user_id, memory_id, content):
        self.updated.append((user_id, memory_id, content))
        return True

    async def delete_memory(self, user_id, memory_id):
        self.deleted.append((user_id, memory_id))
        return True


@pytest.mark.asyncio
async def test_extract_memory_ops_sends_prompt_and_parses():
    with patch.object(ums, "complete_text", new=AsyncMock(return_value='[{"op": "add", "content": "Owns a dog"}]')) as mock_llm:
        ops = await extract_memory_ops("I own a dog named Rex", "That's lovely!", [], backend="ollama", model="gemma3:1b")
    assert ops == [{"op": "add", "content": "Owns a dog"}]
    sent = mock_llm.call_args.args[0]
    assert any(m["role"] == "system" for m in sent)
    assert "I own a dog named Rex" in sent[-1]["content"]
    assert mock_llm.call_args.kwargs["backend"] == "ollama"


@pytest.mark.asyncio
async def test_extract_memory_ops_returns_empty_on_llm_error():
    with patch.object(ums, "complete_text", new=AsyncMock(side_effect=RuntimeError("provider down"))):
        assert await extract_memory_ops("hi", "hello", []) == []


@pytest.mark.asyncio
async def test_apply_memory_ops_dispatches_to_store():
    store = FakeStore()
    await apply_memory_ops(
        store, "user-1",
        [
            {"op": "add", "content": "New fact"},
            {"op": "update", "id": "m1", "content": "Edited"},
            {"op": "delete", "id": "m2"},
        ],
    )
    assert store.added == [("user-1", "New fact", "auto")]
    assert store.updated == [("user-1", "m1", "Edited")]
    assert store.deleted == [("user-1", "m2")]


@pytest.mark.asyncio
async def test_apply_memory_ops_continues_past_a_failing_op(caplog):
    """One failing store call must not drop the ops after it."""

    class FlakyStore(FakeStore):
        def __init__(self):
            super().__init__()
            self._update_calls = 0

        async def update_memory(self, user_id, memory_id, content):
            self._update_calls += 1
            if self._update_calls == 1:
                raise RuntimeError("transient db error")
            return await super().update_memory(user_id, memory_id, content)

    store = FlakyStore()
    with caplog.at_level("WARNING"):
        await apply_memory_ops(
            store, "user-1",
            [
                {"op": "update", "id": "m1", "content": "Fails once"},
                {"op": "delete", "id": "m2"},
            ],
        )
    assert store._update_calls == 1
    assert store.updated == []
    assert store.deleted == [("user-1", "m2")]
    assert any("Memory op update failed" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_run_memory_update_swallows_all_errors():
    with patch.object(ums, "get_user_memory_store", side_effect=RuntimeError("db down")):
        await run_memory_update("user-1", "hi", "hello")  # must not raise


@pytest.mark.asyncio
async def test_run_memory_update_happy_path():
    store = FakeStore(memories=[])
    with patch.object(ums, "get_user_memory_store", return_value=store), \
         patch.object(ums, "complete_text", new=AsyncMock(return_value='[{"op": "add", "content": "Lives in Berlin"}]')):
        await run_memory_update("user-1", "I live in Berlin", "Nice city!")
    assert store.added == [("user-1", "Lives in Berlin", "auto")]


@pytest.mark.asyncio
async def test_load_memory_block_empty_user():
    assert await load_memory_block(None) == ""


@pytest.mark.asyncio
async def test_load_memory_block_degrades_on_store_error():
    with patch.object(ums, "get_user_memory_store", side_effect=RuntimeError("db down")):
        assert await load_memory_block("user-1") == ""


# ── Counter-fact / contradiction handling ──────────────────────

VALID_ID_2 = "7c9e6679-7425-40de-944b-e07fc1f90ae7"


@pytest.mark.asyncio
async def test_extract_memory_ops_updates_contradicted_memory():
    """When the user corrects an existing fact the LLM should emit an update op."""
    existing_mem = type("M", (), {"id": VALID_MEMORY_ID, "content": "Lives in Sydney"})()
    llm_response = f'[{{"op": "update", "id": "{VALID_MEMORY_ID}", "content": "Lives in Melbourne"}}]'
    with patch.object(ums, "complete_text", new=AsyncMock(return_value=llm_response)):
        ops = await extract_memory_ops(
            "I actually moved to Melbourne last month",
            "Good to know, I'll update that!",
            [existing_mem],
            backend="ollama",
            model="gemma3:1b",
        )
    assert len(ops) == 1
    assert ops[0] == {"op": "update", "id": VALID_MEMORY_ID, "content": "Lives in Melbourne"}


@pytest.mark.asyncio
async def test_extract_memory_ops_deletes_and_adds_on_contradiction():
    """Some models emit delete+add instead of update; both are accepted."""
    existing_mem = type("M", (), {"id": VALID_MEMORY_ID, "content": "Prefers Python"})()
    llm_response = (
        f'[{{"op": "delete", "id": "{VALID_MEMORY_ID}"}},'
        f'{{"op": "add", "content": "Prefers Rust"}}]'
    )
    with patch.object(ums, "complete_text", new=AsyncMock(return_value=llm_response)):
        ops = await extract_memory_ops(
            "Actually I switched to Rust full time",
            "Noted, Rust it is!",
            [existing_mem],
        )
    assert len(ops) == 2
    assert ops[0] == {"op": "delete", "id": VALID_MEMORY_ID}
    assert ops[1] == {"op": "add", "content": "Prefers Rust"}


@pytest.mark.asyncio
async def test_run_memory_update_stores_counter_fact_via_update():
    """End-to-end: counter-fact triggers an update that reaches the store."""
    existing_mem = type("M", (), {"id": VALID_MEMORY_ID, "content": "Works at Acme"})()
    store = FakeStore(memories=[existing_mem])
    llm_response = f'[{{"op": "update", "id": "{VALID_MEMORY_ID}", "content": "Works at Globex"}}]'
    with patch.object(ums, "get_user_memory_store", return_value=store), \
         patch.object(ums, "complete_text", new=AsyncMock(return_value=llm_response)):
        await run_memory_update(
            "user-1",
            "I changed jobs, I now work at Globex",
            "Got it, I've updated that.",
        )
    assert store.updated == [("user-1", VALID_MEMORY_ID, "Works at Globex")]
    assert store.added == []


@pytest.mark.asyncio
async def test_extraction_prompt_contains_contradiction_rule():
    """Regression guard: the extraction prompt must explicitly mention contradictions."""
    assert "CONTRADICTION" in ums._EXTRACTION_SYSTEM_PROMPT
    assert "update" in ums._EXTRACTION_SYSTEM_PROMPT.lower()
    assert "delete" in ums._EXTRACTION_SYSTEM_PROMPT.lower()
