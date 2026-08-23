"""Tests for user_memory_service pure functions."""

from app.services.user_memory_service import (
    MAX_MEMORY_OPS_PER_TURN,
    build_memory_block,
    parse_memory_ops,
)


class FakeMemory:
    def __init__(self, content):
        self.content = content


def test_build_memory_block_empty():
    assert build_memory_block([]) == ""


def test_build_memory_block_formats_facts():
    block = build_memory_block([FakeMemory("Prefers concise answers"), FakeMemory("Name is Alex")])
    assert block.startswith("About the user (persistent memory; use when relevant):")
    assert "- Prefers concise answers" in block
    assert "- Name is Alex" in block


def test_parse_memory_ops_valid_json():
    raw = '[{"op": "add", "content": "Likes Rust"}]'
    assert parse_memory_ops(raw) == [{"op": "add", "content": "Likes Rust"}]


def test_parse_memory_ops_strips_code_fence():
    raw = '```json\n[{"op": "delete", "id": "abc"}]\n```'
    assert parse_memory_ops(raw) == [{"op": "delete", "id": "abc"}]


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


def test_parse_memory_ops_truncates_overlong_content():
    raw = '[{"op": "add", "content": "' + "x" * 600 + '"}]'
    ops = parse_memory_ops(raw)
    assert len(ops[0]["content"]) == 500
