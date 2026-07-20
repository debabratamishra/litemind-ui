"""Tests for RAG skill routing across the three supported modes.

Modes (selected by request flags):
  * standard RAG          -> use_multi_agent=False, use_hybrid_search=False
  * hybrid RAG (BM25+vec) -> use_multi_agent=False, use_hybrid_search=True
  * multi-agent RAG       -> use_multi_agent=True  (CrewAI orchestrator)

These tests verify the registry selects the correct skill, and that the
multi-agent skill falls back to standard RAG (with a clear notice) when the
CrewAI dependency is unavailable, rather than silently behaving like standard
RAG without any signal.
"""

from types import SimpleNamespace
from unittest.mock import patch

from app.skills.rag import MultiAgentRAGSkill, StandardRAGSkill
from app.skills.registry import RAGSkillRegistry


def _make_request(**overrides) -> SimpleNamespace:
    base = dict(
        use_multi_agent=False,
        use_hybrid_search=False,
        query="What does the document say about X?",
        system_prompt="You are a helpful assistant.",
        messages=[],
        n_results=3,
        model="gemma3:1b",
        conversation_summary=None,
        backend="ollama",
        api_base=None,
        api_key=None,
        temperature=0.7,
        max_tokens=2048,
        top_p=0.9,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        min_p=0.0,
        seed=None,
        stop=None,
        is_voice_mode=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _registry() -> RAGSkillRegistry:
    fallback = StandardRAGSkill()
    return RAGSkillRegistry(
        skills=[
            MultiAgentRAGSkill(fallback_skill=fallback),
            fallback,
        ]
    )


def test_registry_resolves_multi_agent_when_flag_set():
    skill = _registry().resolve(_make_request(use_multi_agent=True))
    assert isinstance(skill, MultiAgentRAGSkill)


def test_registry_resolves_standard_for_both_hybrid_states():
    # StandardRAGSkill handles both standard (hybrid off) and hybrid (hybrid on)
    # queries; multi-agent is only selected when use_multi_agent is True.
    for hybrid in (False, True):
        skill = _registry().resolve(_make_request(use_multi_agent=False, use_hybrid_search=hybrid))
        assert isinstance(skill, StandardRAGSkill)


def test_supports_gating():
    assert StandardRAGSkill().supports(_make_request(use_multi_agent=False)) is True
    assert StandardRAGSkill().supports(_make_request(use_multi_agent=True)) is False
    assert MultiAgentRAGSkill(fallback_skill=StandardRAGSkill()).supports(_make_request(use_multi_agent=True)) is True
    assert MultiAgentRAGSkill(fallback_skill=StandardRAGSkill()).supports(_make_request(use_multi_agent=False)) is False


async def test_multi_agent_falls_back_when_crewai_unavailable():
    async def fake_query(*_args, **_kwargs):
        yield "STANDARD_ANSWER"

    rag_service = SimpleNamespace(query=fake_query)
    skill = MultiAgentRAGSkill(fallback_skill=StandardRAGSkill())
    req = _make_request(use_multi_agent=True, use_hybrid_search=False)

    with patch(
        "app.services.rag_multi_agent.multi_agent_rag_available",
        return_value=(False, "No module named 'crewai'"),
    ):
        chunks = [c async for c in skill.stream(req, rag_service)]

    # A clear fallback notice must precede the standard-RAG output so the user
    # knows multi-agent did not run.
    assert any("Falling back to standard RAG" in c for c in chunks)
    # The actual answer still comes through via the standard skill.
    assert any("STANDARD_ANSWER" in c for c in chunks)
