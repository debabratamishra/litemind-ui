"""Unit tests for the pluggable skill layer (base, registry, rag, web_search).

These tests run fully offline: no network, no real LLM/web-search providers.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import AsyncIterator

from app.skills.base import SkillValidationResult, StreamingChatSkill
from app.skills.rag import MultiAgentRAGSkill, StandardRAGSkill
from app.skills.registry import ChatSkillRegistry, RAGSkillRegistry
from app.skills.web_search import (
    WebSearchChatSkill,
    build_web_search_conversation_history,
)


# ──────────────────────────────────────────────────────────────────────────
# base: SkillValidationResult
# ──────────────────────────────────────────────────────────────────────────
def test_skill_validation_result_defaults():
    ok = SkillValidationResult(ok=True)
    assert ok.ok is True
    assert ok.message is None

    bad = SkillValidationResult(ok=False, message="nope")
    assert bad.ok is False and bad.message == "nope"


def test_skill_validation_result_is_frozen():
    result = SkillValidationResult(ok=True)
    try:
        result.ok = False  # type: ignore[misc]
    except (AttributeError, dataclasses.FrozenInstanceError):
        # dataclass(frozen=True) raises a generated error; both are acceptable.
        return
    raise AssertionError("SkillValidationResult should be immutable")


# ──────────────────────────────────────────────────────────────────────────
# Minimal fake skill implementing the StreamingChatSkill protocol.
# ──────────────────────────────────────────────────────────────────────────
class FakeChatSkill:
    name = "fake"
    description = "A fake chat skill for registry tests."

    def supports(self, request: object) -> bool:
        return getattr(request, "mode", None) == "fake"

    def validate(self, request: object) -> SkillValidationResult:
        return SkillValidationResult(ok=True)

    async def stream(self, request: object) -> AsyncIterator[str]:
        yield "x"


class OtherFakeChatSkill:
    name = "other"
    description = "Another fake chat skill."

    def supports(self, request: object) -> bool:
        return getattr(request, "mode", None) == "other"

    def validate(self, request: object) -> SkillValidationResult:
        return SkillValidationResult(ok=True)

    async def stream(self, request: object) -> AsyncIterator[str]:
        yield "y"


def _fake_request(mode: str) -> SimpleNamespace:
    return SimpleNamespace(mode=mode)


# ──────────────────────────────────────────────────────────────────────────
# registry: ChatSkillRegistry
# ──────────────────────────────────────────────────────────────────────────
def test_chat_registry_resolves_first_match():
    reg = ChatSkillRegistry([FakeChatSkill(), OtherFakeChatSkill()])

    assert reg.resolve(_fake_request("fake")) is reg.skills[0]
    assert reg.resolve(_fake_request("other")) is reg.skills[1]
    # No registered skill supports an unknown mode.
    assert reg.resolve(_fake_request("unknown")) is None
    assert reg.resolve(object()) is None


def test_chat_registry_skills_immutable():
    reg = ChatSkillRegistry([FakeChatSkill()])
    assert isinstance(reg.skills, tuple)


def test_chat_registry_streams_through_resolved_skill():
    reg = ChatSkillRegistry([FakeChatSkill()])

    async def run() -> None:
        skill = reg.resolve(_fake_request("fake"))
        assert skill is not None
        chunks = [chunk async for chunk in skill.stream(_fake_request("fake"))]
        assert chunks == ["x"]

    import anyio

    anyio.run(run)


# ──────────────────────────────────────────────────────────────────────────
# registry: RAGSkillRegistry
# ──────────────────────────────────────────────────────────────────────────
def test_rag_registry_get_by_name():
    std = StandardRAGSkill()
    multi = MultiAgentRAGSkill(fallback_skill=std)
    reg = RAGSkillRegistry([std, multi])

    assert reg.get("standard_rag") is std
    assert reg.get("multi_agent_rag") is multi
    # Unknown name returns None.
    assert reg.get("does_not_exist") is None


def test_rag_registry_resolve_by_support():
    std = StandardRAGSkill()
    multi = MultiAgentRAGSkill(fallback_skill=std)
    reg = RAGSkillRegistry([std, multi])

    plain = SimpleNamespace(use_multi_agent=False)
    agentic = SimpleNamespace(use_multi_agent=True)

    assert reg.resolve(plain) is std
    assert reg.resolve(agentic) is multi
    # use_multi_agent=None is falsy, so standard RAG claims it.
    assert reg.resolve(SimpleNamespace(use_multi_agent=None)) is std


# ──────────────────────────────────────────────────────────────────────────
# rag: StandardRAGSkill vs MultiAgentRAGSkill
# ──────────────────────────────────────────────────────────────────────────
def test_standard_rag_skill_attributes_and_support():
    std = StandardRAGSkill()
    assert std.name == "standard_rag"
    assert std.description
    # Standard supports requests that do NOT ask for multi-agent.
    assert std.supports(SimpleNamespace(use_multi_agent=False)) is True
    assert std.supports(SimpleNamespace(use_multi_agent=True)) is False


def test_multiagent_rag_skill_requires_fallback_and_support():
    std = StandardRAGSkill()
    multi = MultiAgentRAGSkill(fallback_skill=std)
    assert multi.name == "multi_agent_rag"
    assert multi.description
    # Multi-agent supports requests that DO ask for multi-agent.
    assert multi.supports(SimpleNamespace(use_multi_agent=True)) is True
    assert multi.supports(SimpleNamespace(use_multi_agent=False)) is False


def test_standard_vs_multiagent_rag_distinct():
    std = StandardRAGSkill()
    multi = MultiAgentRAGSkill(fallback_skill=std)
    assert std is not multi
    assert std.name != multi.name


# ──────────────────────────────────────────────────────────────────────────
# web_search: WebSearchChatSkill + build_web_search_conversation_history
# ──────────────────────────────────────────────────────────────────────────
def test_web_search_chat_skill_attributes_and_support():
    skill = WebSearchChatSkill()
    assert skill.name == "web_search"
    assert skill.description
    # supports() only inspects the request flag; no network involved.
    assert skill.supports(SimpleNamespace(use_web_search=True)) is True
    assert skill.supports(SimpleNamespace(use_web_search=False)) is False


def test_build_web_search_conversation_history_empty():
    request = SimpleNamespace(conversation_summary=None, conversation_history=[])
    history = build_web_search_conversation_history(request)
    assert history == []


def test_build_web_search_conversation_history_summary_and_messages():
    request = SimpleNamespace(
        conversation_summary="Prior context summary.",
        conversation_history=[
            SimpleNamespace(role="user", content="Hello"),
            SimpleNamespace(role="assistant", content="Hi there"),
        ],
    )
    history = build_web_search_conversation_history(request)

    assert isinstance(history, list)
    # Summary is prepended as a system message.
    assert history[0] == {
        "role": "system",
        "content": "Summary of previous conversation:\nPrior context summary.",
    }
    # Then the flattened conversation history.
    assert history[1:] == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
