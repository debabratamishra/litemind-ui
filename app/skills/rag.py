"""RAG strategy skills for standard and multi-agent query flows."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from app.services.rag_multi_agent import CrewAIRAGOrchestrator, multi_agent_rag_available

logger = logging.getLogger(__name__)


def _rag_messages(request: Any) -> list[dict[str, Any]]:
    """Copy request messages into a plain list for downstream services."""

    return request.messages.copy() if request.messages else []


class StandardRAGSkill:
    """Default retrieval and generation flow for RAG requests."""

    name = "standard_rag"
    description = "Use the standard retrieval pipeline and stream the answer."

    def supports(self, request: Any) -> bool:
        return not bool(getattr(request, "use_multi_agent", False))

    async def stream(self, request: Any, rag_service: Any) -> AsyncIterator[str]:
        async for chunk in rag_service.query(
            request.query,
            request.system_prompt,
            _rag_messages(request),
            request.n_results,
            request.use_hybrid_search,
            request.model,
            conversation_summary=request.conversation_summary,
            backend=request.backend,
            api_base=request.api_base,
            api_key=request.api_key,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            frequency_penalty=request.frequency_penalty,
            repetition_penalty=request.repetition_penalty,
            is_voice_mode=request.is_voice_mode,
        ):
            yield chunk


class MultiAgentRAGSkill:
    """CrewAI-backed RAG strategy with built-in standard RAG fallback."""

    name = "multi_agent_rag"
    description = "Use CrewAI to refine the query and compose the final RAG answer."

    def __init__(self, fallback_skill: StandardRAGSkill):
        self._fallback_skill = fallback_skill

    def supports(self, request: Any) -> bool:
        return bool(getattr(request, "use_multi_agent", False))

    async def stream(self, request: Any, rag_service: Any) -> AsyncIterator[str]:
        available, detail = multi_agent_rag_available()
        if not available:
            logger.warning("Multi-agent RAG unavailable, falling back to standard RAG: %s", detail)
            yield self._fallback_message(detail)
            async for chunk in self._fallback_skill.stream(request, rag_service):
                yield chunk
            return

        try:
            orchestrator = CrewAIRAGOrchestrator(
                rag_service=rag_service,
                model_name=request.model or "gemma3:1b",
                backend=request.backend,
                api_base=request.api_base,
                api_key=request.api_key,
            )
        except RuntimeError as exc:
            detail = str(exc)
            logger.warning("Multi-agent RAG initialization failed, falling back: %s", detail)
            yield self._fallback_message(detail)
            async for chunk in self._fallback_skill.stream(request, rag_service):
                yield chunk
            return

        async for chunk in orchestrator.query(
            user_query=request.query,
            system_prompt=request.system_prompt,
            messages=_rag_messages(request),
            n_results=request.n_results,
            use_hybrid_search=request.use_hybrid_search,
            model=request.model,
            conversation_summary=request.conversation_summary,
        ):
            yield chunk

    @staticmethod
    def _fallback_message(detail: str | None) -> str:
        message = "Multi-agent orchestration is not available in this environment."
        if detail:
            message += f" {detail}"
        return message + " Falling back to standard RAG.\n\n"
