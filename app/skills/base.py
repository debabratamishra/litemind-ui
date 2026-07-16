"""Shared types for pluggable chat and RAG skills."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator, Protocol

if TYPE_CHECKING:
    from app.backend.models.api_models import ChatRequestEnhanced


@dataclass(frozen=True)
class SkillValidationResult:
    """Outcome of capability-specific request validation."""

    ok: bool
    message: str | None = None


class StreamingChatSkill(Protocol):
    """Protocol for chat skills that can stream responses."""

    name: str
    description: str

    def supports(self, request: ChatRequestEnhanced) -> bool:
        """Return True when this skill should handle the request."""

    def validate(self, request: ChatRequestEnhanced) -> SkillValidationResult:
        """Validate request prerequisites before streaming begins."""

    def stream(self, request: ChatRequestEnhanced) -> AsyncIterator[str]:
        """Yield streamed response chunks for the request."""


class StreamingRAGSkill(Protocol):
    """Protocol for RAG skills that can stream responses."""

    name: str
    description: str

    def supports(self, request: Any) -> bool:
        """Return True when this skill should handle the request."""

    def stream(self, request: Any, rag_service: Any) -> AsyncIterator[str]:
        """Yield streamed response chunks for the request."""
