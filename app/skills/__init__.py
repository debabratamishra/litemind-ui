"""Pluggable backend skills for chat and RAG capabilities."""

from typing import cast

from app.skills.base import StreamingChatSkill, StreamingRAGSkill
from app.skills.rag import MultiAgentRAGSkill, StandardRAGSkill
from app.skills.registry import ChatSkillRegistry, RAGSkillRegistry
from app.skills.web_search import WebSearchChatSkill

chat_skills = cast(
    "list[StreamingChatSkill]",
    [WebSearchChatSkill()],
)

chat_skill_registry = ChatSkillRegistry(
    skills=chat_skills
)

_standard_rag_skill = StandardRAGSkill()

rag_skills = cast(
    "list[StreamingRAGSkill]",
    [
        MultiAgentRAGSkill(fallback_skill=_standard_rag_skill),
        _standard_rag_skill,
    ],
)

rag_skill_registry = RAGSkillRegistry(
    skills=rag_skills
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
