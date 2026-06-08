"""Registries for resolving pluggable chat and RAG skills."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

from app.skills.base import StreamingChatSkill, StreamingRAGSkill

if TYPE_CHECKING:
    from app.backend.models.api_models import ChatRequestEnhanced


class ChatSkillRegistry:
    """Resolve the first registered skill that supports a request."""

    def __init__(self, skills: Iterable[StreamingChatSkill]):
        self._skills = tuple(skills)

    @property
    def skills(self) -> tuple[StreamingChatSkill, ...]:
        return self._skills

    def resolve(self, request: ChatRequestEnhanced) -> StreamingChatSkill | None:
        for skill in self._skills:
            if skill.supports(request):
                return skill
        return None


class RAGSkillRegistry:
    """Resolve the first registered RAG skill that supports a request."""

    def __init__(self, skills: Iterable[StreamingRAGSkill]):
        self._skills = tuple(skills)

    @property
    def skills(self) -> tuple[StreamingRAGSkill, ...]:
        return self._skills

    def get(self, name: str) -> StreamingRAGSkill | None:
        for skill in self._skills:
            if skill.name == name:
                return skill
        return None

    def resolve(self, request: Any) -> StreamingRAGSkill | None:
        for skill in self._skills:
            if skill.supports(request):
                return skill
        return None
