"""Web search orchestrator using direct LLM prompting and SerpAPI results."""

import logging
from typing import AsyncGenerator, Dict, List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

from app.services.llm_gateway import normalize_backend, resolve_backend_config, stream_completion
from app.services.web_search_service import WebSearchService

logger = logging.getLogger(__name__)


class WebSearchOrchestrator:
    """Orchestrates web search retrieval and synthesis."""

    def __init__(
        self,
        backend: str = "ollama",
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize the orchestrator with WebSearchService and a configurable LLM."""
        self.web_search_service = WebSearchService()

        self.backend = normalize_backend(backend)
        self.requested_model = model
        self.llm_config = resolve_backend_config(
            backend=self.backend,
            model=model,
            api_base=api_base,
            api_key=api_key,
        )

        logger.info(
            "WebSearchOrchestrator initialized with backend=%s model=%s api_base=%s",
            self.backend,
            self.llm_config.model,
            self.llm_config.api_base,
        )

    async def process_query(
        self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None, stream: bool = True
    ) -> AsyncGenerator[str, None]:
        """Process a user query through the web search workflow.

        This is the main entry point for web search queries. It orchestrates:
        1. Search execution via SerpAPI
        2. Result synthesis via the configured LLM
        3. Streaming response generation

        Args:
            query: The user's search query
            conversation_history: Optional list of previous messages for context
            stream: Whether to stream the response (default: True)

        Yields:
            str: Chunks of the synthesized response

        Raises:
            Exception: Falls back to Base LLM on any error
        """
        try:
            logger.info(f"Processing web search query: '{query}'")

            # Step 1: Optimize the search query
            optimized_query = await self._query_optimizer_process(query)
            logger.info(f"Optimized query: '{optimized_query}'")

            # Step 2: Execute search using the optimized query
            search_results = await self._serper_agent_search(optimized_query)

            if not search_results:
                logger.warning("No search results retrieved, falling back to Base LLM")
                async for chunk in self._fallback_to_base_llm(query, conversation_history):
                    yield chunk
                return

            # Step 3: Synthesize the search results
            async for chunk in self._synthesizer_agent_process(
                query=query,  # Use original query for context
                optimized_query=optimized_query,
                search_results=search_results,
                conversation_history=conversation_history,
                stream=stream,
            ):
                yield chunk

        except Exception as e:
            logger.error(f"Error in web search workflow: {e}", exc_info=True)
            logger.info("Falling back to Base LLM due to error")

            # Fallback to Base LLM on any error
            async for chunk in self._fallback_to_base_llm(query, conversation_history):
                yield chunk

    async def _query_optimizer_process(self, query: str) -> str:
        """Optimize the user query for better search results.

        Args:
            query: The original user query

        Returns:
            str: Optimized search query
        """
        try:
            logger.info(f"Query Optimizer processing: '{query}'")

            optimization_prompt = (
                f"Optimize the following user question into a clear, focused search query. "
                f"Extract key terms, remove unnecessary words, and create a query that will "
                f"retrieve the most relevant search results. Return ONLY the optimized query, "
                f"nothing else.\n\n"
                f"Original question: {query}\n\n"
                f"Optimized search query:"
            )

            # Get optimized query from LLM
            optimized_query = ""
            async for chunk in self._stream_from_llm(optimization_prompt):
                optimized_query += chunk

            # Clean up the response - remove any extra text
            optimized_query = optimized_query.strip()

            # If optimization failed or is too short, use original
            if not optimized_query or len(optimized_query) < 3:
                logger.warning("Query optimization failed, using original query")
                return query

            return optimized_query

        except Exception as e:
            logger.error(f"Query optimization failed: {e}", exc_info=True)
            return query  # Fallback to original query

    async def _serper_agent_search(self, query: str) -> List[Dict[str, any]]:
        """Execute search via WebSearchService and return formatted results.

        Args:
            query: The search query string

        Returns:
            list: Formatted search results (up to 10 results)

        Raises:
            Exception: If search fails
        """
        try:
            logger.info(f"Executing web search for: '{query}'")

            # Execute search via WebSearchService (retrieves exactly 10 results)
            raw_results = await self.web_search_service.search(query, num_results=10)

            # Format results for agent consumption
            formatted_results = self.web_search_service.format_results(raw_results)

            logger.info(f"Retrieved {len(formatted_results)} search results")
            return formatted_results

        except ValueError as e:
            # API key validation error
            logger.error(f"SerpAPI token validation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Search execution failed: {e}", exc_info=True)
            raise

    async def _synthesizer_agent_process(
        self,
        query: str,
        optimized_query: str,
        search_results: List[Dict[str, any]],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        stream: bool = True,
    ) -> AsyncGenerator[str, None]:
        """Synthesize search results into a comprehensive response using Base LLM.

        Args:
            query: The original user query
            optimized_query: The optimized search query used
            search_results: Formatted search results from the search backend
            conversation_history: Optional conversation context
            stream: Whether to stream the response

        Yields:
            str: Chunks of the synthesized response
        """
        try:
            logger.info("Synthesizing web search results")

            # Build context from search results
            search_context = self._build_search_context(search_results)

            # Build conversation context if available
            conversation_context = ""
            if conversation_history:
                conversation_context = self._build_conversation_context(conversation_history)

            # Create synthesis prompt
            synthesis_prompt = self._create_synthesis_prompt(
                query=query,
                optimized_query=optimized_query,
                search_context=search_context,
                conversation_context=conversation_context,
            )

            # Stream response from Base LLM
            logger.info("Streaming synthesis response from Base LLM")
            async for chunk in self._stream_from_llm(synthesis_prompt):
                yield chunk

            citation_block = self._build_citation_block(search_results)
            if citation_block:
                yield "\n\nSources:\n" + citation_block

        except Exception as e:
            logger.error(f"Synthesis failed: {e}", exc_info=True)
            raise

    def _build_search_context(self, search_results: List[Dict[str, any]]) -> str:
        """Build a formatted context string from search results.

        Args:
            search_results: List of formatted search results

        Returns:
            str: Formatted search context for the synthesis prompt
        """
        if not search_results:
            return "No search results available."

        context_parts = ["Here are the top search results:\n"]

        for result in search_results:
            position = result.get("position", "?")
            title = result.get("title", "No title")
            link = result.get("link", "")
            snippet = result.get("snippet", "No description")

            context_parts.append(f"\n[{position}] {title}\nURL: {link}\nDescription: {snippet}\n")

        return "".join(context_parts)

    def _build_citation_block(self, search_results: List[Dict[str, any]]) -> str:
        """Create a clean, well-formatted citation block from search results.

        Format each source as:
        [N] **Title** (domain) - [Link](url)
            *Brief description/snippet*
        """
        if not search_results:
            return ""

        lines = []
        for result in search_results:
            link = (result.get("link") or "").strip()
            title = result.get("title", "Source").strip()
            position = result.get("position", "?")
            snippet = result.get("snippet", "").strip()

            if not link:
                continue

            # Extract domain from URL
            domain = urlparse(link).netloc or link
            # Remove www. prefix for cleaner display
            if domain.startswith("www."):
                domain = domain[4:]

            # Build the citation line with markdown formatting
            # Format: [N] **Title** (domain) - [Link](url)
            citation_line = f"[{position}] **{title}** ({domain}) - [Link]({link})"

            # Add snippet as italicized text on next line if available
            if snippet:
                # Truncate long snippets
                if len(snippet) > 200:
                    snippet = snippet[:197] + "..."
                citation_line += f"\n    *{snippet}*"

            lines.append(citation_line)

        return "\n\n".join(lines)

    def _build_conversation_context(self, conversation_history: List[Dict[str, str]]) -> str:
        """Build conversation context from message history.

        Args:
            conversation_history: List of previous messages

        Returns:
            str: Formatted conversation context
        """
        if not conversation_history:
            return ""

        context_parts = ["Previous conversation:\n"]

        # Include last few messages for context (limit to avoid token overflow)
        recent_messages = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history

        for msg in recent_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            context_parts.append(f"{role.capitalize()}: {content}\n")

        return "".join(context_parts)

    def _create_synthesis_prompt(
        self, query: str, optimized_query: str, search_context: str, conversation_context: str
    ) -> str:
        """Create the synthesis prompt for the Base LLM.

        Args:
            query: The user's original query
            optimized_query: The optimized search query used
            search_context: Formatted search results
            conversation_context: Formatted conversation history

        Returns:
            str: Complete synthesis prompt
        """
        prompt_parts = []

        # Add conversation context if available
        if conversation_context:
            prompt_parts.append(conversation_context)
            prompt_parts.append("\n---\n")

        # Add search context
        prompt_parts.append(search_context)
        prompt_parts.append("\n---\n")

        # Detailed prompt for natural synthesis with proper citation placement
        prompt_parts.append(
            f"Using the search results above, answer this question: {query}\n\n"
            f"IMPORTANT INSTRUCTIONS:\n"
            f"1. Write complete, natural sentences. Do NOT start sentences with citation numbers.\n"
            f"2. Place citation numbers [1], [2], etc. at the END of the sentence or claim they support.\n"
            f"3. Write in plain text only - no markdown (no *, **, #, bullets).\n"
            f"4. Synthesize information into your own words - do not just quote the sources.\n"
            f"5. Be specific with facts and figures rather than vague references.\n\n"
            f"CORRECT citation style:\n"
            f'- "The weather will be partly cloudy with temperatures around 25°C [3]."\n'
            f'- "Rain is expected on Tuesday and Wednesday [1][4]."\n\n'
            f'- "The stock price was up 0.04% relative to the opening price today [5]\n\n'
            f"INCORRECT citation style (do NOT do this):\n"
            f'- "[2] and [7] both mention..." (Never start with citations)\n'
            f'- "According to [3]..." (Avoid this pattern)\n\n'
            f'- "[5] indicates that the stock price was up 0.04% relative to the opening price today [5]" (Avoid this pattern)\n\n'
            f'- "[5] and [6] both indicate positive price projections."\n\n (Never start with citations)\n'
            f"Answer:"
        )

        return "".join(prompt_parts)

    async def _stream_from_llm(self, prompt: str) -> AsyncGenerator[str, None]:
        """Stream response from the configured base LLM.

        Args:
            prompt: The complete prompt to send to the model

        Yields:
            str: Response chunks from the LLM
        """
        try:
            async for chunk in stream_completion(
                [{"role": "user", "content": prompt}],
                backend=self.backend,
                model=self.requested_model,
                api_base=self.llm_config.api_base,
                api_key=self.llm_config.api_key,
                temperature=0.2,
                max_tokens=2048,
            ):
                yield chunk
        except Exception as e:
            logger.error(f"Error streaming from configured LLM: {e}", exc_info=True)
            raise

    async def _fallback_to_base_llm(
        self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> AsyncGenerator[str, None]:
        """Fallback to Base LLM when web search fails.

        Args:
            query: The user's query
            conversation_history: Optional conversation context

        Yields:
            str: Response chunks from Base LLM
        """
        logger.info("Using Base LLM fallback (no web search)")

        # Build simple prompt without search results
        prompt_parts = []

        if conversation_history:
            conversation_context = self._build_conversation_context(conversation_history)
            prompt_parts.append(conversation_context)
            prompt_parts.append("\n---\n")

        prompt_parts.append(f"User: {query}\n\nAssistant: ")

        fallback_prompt = "".join(prompt_parts)

        async for chunk in self._stream_from_llm(fallback_prompt):
            yield chunk
