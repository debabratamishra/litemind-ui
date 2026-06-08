"""Pluggable backend skills for chat and RAG capabilities."""

from app.skills.rag import MultiAgentRAGSkill, StandardRAGSkill
from app.skills.registry import ChatSkillRegistry, RAGSkillRegistry
from app.skills.web_search import WebSearchChatSkill

chat_skill_registry = ChatSkillRegistry(
    skills=[
        WebSearchChatSkill(),
    ]
)

_standard_rag_skill = StandardRAGSkill()

rag_skill_registry = RAGSkillRegistry(
    skills=[
        MultiAgentRAGSkill(fallback_skill=_standard_rag_skill),
        _standard_rag_skill,
    ]
)

__all__ = [
    "ChatSkillRegistry",
    "RAGSkillRegistry",
    "WebSearchChatSkill",
    "StandardRAGSkill",
    "MultiAgentRAGSkill",
    "chat_skill_registry",
    "rag_skill_registry",
]
