"""Web search skill implementation for chat requests."""

from __future__ import annotations

from typing import TYPE_CHECKING, AsyncIterator

from app.services.web_search_crew import WebSearchOrchestrator
from app.services.web_search_service import WebSearchService
from app.skills.base import SkillValidationResult

if TYPE_CHECKING:
    from app.backend.models.api_models import ChatRequestEnhanced


def build_web_search_conversation_history(
    request: ChatRequestEnhanced,
) -> list[dict[str, str]]:
    """Normalize chat request memory fields into skill conversation history."""

    conversation_history: list[dict[str, str]] = []

    if request.conversation_summary:
        conversation_history.append(
            {
                "role": "system",
                "content": f"Summary of previous conversation:\n{request.conversation_summary}",
            }
        )

    if request.conversation_history:
        conversation_history.extend(
            {"role": message.role, "content": message.content} for message in request.conversation_history
        )

    return conversation_history


class WebSearchChatSkill:
    """Skill for answering chat requests using web search."""

    name = "web_search"
    description = "Search the web, synthesize results, and stream the answer."

    def supports(self, request: ChatRequestEnhanced) -> bool:
        return bool(request.use_web_search)

    def validate(self, request: ChatRequestEnhanced) -> SkillValidationResult:
        validation = WebSearchService(api_key=request.serp_api_key).validate_token()
        if validation["valid"]:
            return SkillValidationResult(ok=True)

        return SkillValidationResult(ok=False, message=validation["message"])

    async def stream(self, request: ChatRequestEnhanced) -> AsyncIterator[str]:
        orchestrator = WebSearchOrchestrator(
            backend=request.backend,
            model=request.model,
            api_base=request.api_base,
            api_key=request.api_key,
            serp_api_key=request.serp_api_key,
        )

        async for chunk in orchestrator.process_query(
            query=request.message,
            conversation_history=build_web_search_conversation_history(request),
            stream=True,
        ):
            yield chunk
