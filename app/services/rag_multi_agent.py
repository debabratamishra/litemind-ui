"""Multi-agent RAG orchestration via CrewAI — extracted from rag_service.py."""

import asyncio
import json
import logging
from typing import Any, List, Optional

from app.services.llm_gateway import resolve_backend_config

logger = logging.getLogger(__name__)


def multi_agent_rag_available() -> tuple[bool, str]:
    """Return (True, "ok") if CrewAI and crewai.run types are loadable."""
    try:
        from crewai import Agent, Crew, Process, Task
        from crewai.tools.base_tool import Tool
        return True, "ok"
    except ImportError as e:
        return False, str(e)


# Shared storage for retrieval records — populated by the tool, read by the orchestrator
_last_retrieval_records: List[dict] = []


def _load_crewai_types():
    """Deferred crewai import to avoid hard dependency."""
    from crewai import Agent, Crew, LLM, Process, Task
    from crewai.tools.base_tool import Tool, tool
    return Agent, LLM, Task, Crew, Process, Tool, tool


def _build_retrieve_tool(rag_service, tool):
    """Factory that builds a CrewAI @tool decorator-based retrieval tool.

    Uses the @tool decorator instead of Tool(func=...) to ensure CrewAI correctly
    passes Pydantic-validated arguments to the function.
    """
    global _last_retrieval_records

    @tool("retrieve_knowledge")
    def retrieve_knowledge(query: str, n_results: int = 3) -> str:
        """Search the ingested document knowledge base for the most relevant chunks
        to a user question. Returns top results with source and relevance score.
        Use when you need specific facts, explanations, or context from uploaded documents.

        Args:
            query: The search query to find relevant document chunks.
            n_results: Maximum number of result chunks to return.
        """
        try:
            collection = rag_service.text_collection
        except Exception as e:
            logger.warning("retrieve_knowledge: failed to get collection: %s", e)
            return "[No results - knowledge base unavailable]"

        try:
            count = collection.count()
        except Exception as e:
            logger.warning("retrieve_knowledge: failed to count collection: %s", e)
            return "[No results - knowledge base unavailable]"

        if count == 0:
            return "[No results]"

        try:
            results = collection.query(query_texts=[query], n_results=n_results)
        except Exception as e:
            logger.warning("retrieve_knowledge: query failed: %s", e)
            return "[No results]"

        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not docs:
            return "[No results]"

        chunks = []
        for doc, meta, dist in zip(docs, metadatas, distances):
            source = (meta or {}).get("source", "unknown")
            page_num = (meta or {}).get("page_number")
            score = 1.0 - dist if dist is not None else 0.0
            chunks.append(f"[Source: {source} | Relevance: {score:.2f}]\n{doc}")

        # Capture records for citations metadata (used by frontend for clickable citations)
        global _last_retrieval_records
        _last_retrieval_records = []
        for doc, meta, dist in zip(docs, metadatas, distances):
            if doc:
                source = (meta or {}).get("source", "unknown")
                page_num = (meta or {}).get("page_number")
                score = 1.0 - dist if dist is not None else 0.0
                _last_retrieval_records.append({
                    "id": source,
                    "content": doc,
                    "score": score,
                    "retrieval_method": "multi-agent retrieval",
                    "metadata": {"filename": source, "page_number": page_num} if page_num else {"filename": source},
                })

        return "\n\n---\n\n".join(chunks)

    return retrieve_knowledge


class CrewAIRAGOrchestrator:
    """Proper multi-agent RAG orchestration using CrewAI with a real Crew, sequential agents, and a retrieval tool.

    Architecture:
      researcher  →  synthesizer  →  yield streamed chunks
                   ↑
             (reads researcher output from context)

    All agents collaborate through a shared Crew.kickoff_async() call so CrewAI
    manages task routing, context passing, and delegation natively.
    """

    def __init__(
        self,
        rag_service,
        model_name: str = "gemma3:1b",
        backend: str = "ollama",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Configure the CrewAI LLM, agents, retrieval tool, and Crew."""
        Agent, LLM, Task, Crew, Process, Tool, tool = _load_crewai_types()
        self.Crew = Crew
        self.Task = Task
        self.rag_service = rag_service
        self.llm_config = resolve_backend_config(
            backend=backend, model=model_name, api_base=api_base, api_key=api_key
        )
        self.backend = self.llm_config.backend
        self.model_name = self.llm_config.model

        # Normalise model string for CrewAI LLM
        agent_model = self.model_name
        if self.backend == "ollama" and agent_model.startswith("ollama_chat/"):
            agent_model = f"ollama/{agent_model[len('ollama_chat/'):]}"

        agent_llm = LLM(
            model=agent_model,
            base_url=self.llm_config.api_base,
            api_key=self.llm_config.api_key,
            temperature=0.0,
        )

        # ------------------------------------------------------------------
        # Retrieval tool — gives the researcher agent real ChromaDB query power
        # Uses @tool decorator for proper CrewAI argument binding
        # ------------------------------------------------------------------
        self.retrieval_tool = _build_retrieve_tool(rag_service, tool)

        # ------------------------------------------------------------------
        # Agent 1 — Researcher: understands intent and retrieves docs
        # ------------------------------------------------------------------
        self.researcher = Agent(
            role="Research Analyst",
            goal=(
                "Find the most relevant information from the knowledge base to answer the user's question. "
                "Always use the retrieve_knowledge tool to search the document store before summarising."
            ),
            backstory=(
                "You are a thorough research analyst. Given a question, you break it into "
                "searchable concepts and retrieve supporting evidence from the ingested documents."
            ),
            llm=agent_llm,
            tools=[self.retrieval_tool],
            verbose=True,
        )

        # ------------------------------------------------------------------
        # Agent 2 — Synthesiser: composes the final, cited answer
        # ------------------------------------------------------------------
        self.synthesiser = Agent(
            role="Answer Synthesiser",
            goal=(
                "Produce a clear, well-structured answer that directly answers the user's question "
                "using the research findings provided in context. Cite sources as [1], [2], etc. "
                "within the answer body — never in a separate section."
            ),
            backstory=(
                "You synthesise complex research into accurate, readable answers. "
                "You always ground your response in the provided context and cite correctly."
            ),
            llm=agent_llm,
            verbose=True,
        )

        # ------------------------------------------------------------------
        # Tasks
        # ------------------------------------------------------------------
        self.research_task = Task(
            description=(
                "Given the user question below, use the retrieve_knowledge tool to find "
                "the most relevant chunks. Then produce a concise research summary of what you found, "
                "including any key facts, definitions, or statements that are relevant to the question.\n\n"
                "IMPORTANT: When calling retrieve_knowledge, set n_results={n_results} to get exactly that many chunks.\n\n"
                "User question: {user_query}"
            ),
            expected_output=(
                "A concise research summary: key facts and relevant findings from the knowledge base, "
                "formatted as bullet points or short paragraphs."
            ),
            agent=self.researcher,
        )

        self.synthesis_task = Task(
            description=(
                "Using the research findings already produced in the previous step, write a clear and "
                "complete answer to the original user question. Ground every factual claim in the "
                "research findings and cite sources inline as [1], [2], etc. Do not invent information "
                "not present in the research.\n\n"
                "User question: {user_query}\n"
                "System instructions to honour: {system_prompt}"
            ),
            expected_output=(
                "A well-structured, cited answer to the user question, with sources noted inline. "
                "No separate references section."
            ),
            agent=self.synthesiser,
        )

        # ------------------------------------------------------------------
        # Crew — sequential so synthesizer reads researcher output from context
        # ------------------------------------------------------------------
        self.crew = Crew(
            agents=[self.researcher, self.synthesiser],
            tasks=[self.research_task, self.synthesis_task],
            process=Process.sequential,
            verbose=True,
        )

    # ------------------------------------------------------------------
    # Query — kick off the crew and stream the answer
    # ------------------------------------------------------------------
    async def query(
        self,
        user_query: str,
        system_prompt: str,
        messages: list = [],
        n_results: int = 3,
        use_hybrid_search: bool = False,
        model: Optional[str] = None,
        conversation_summary: Optional[str] = None,
    ):
        """Run the crew synchronously in a thread executor and stream tokens from the final output."""
        global _last_retrieval_records

        logger.info("=== Multi-agent RAG ===")
        logger.info("  User query : %s", user_query)
        logger.info("  Model      : %s (%s)", self.model_name, self.backend)
        logger.info("  n_results  : %d", n_results)
        logger.info("  crew agents: %s", [a.role for a in self.crew.agents])

        # Build conversation context for the agents
        history_parts = []
        if conversation_summary:
            history_parts.append(f"Summary of previous conversation:\n{conversation_summary}")
        history_parts.extend(
            f"{msg['role']}: {msg['content']}" for msg in messages if msg.get("role") != "system"
        )
        history_context = "\n".join(history_parts)
        if history_context:
            logger.info("  history context: %s", history_context[:200])

        crew_inputs = {
            "user_query": user_query,
            "system_prompt": system_prompt,
            "history_context": history_context,
            "n_results": n_results,
        }
        logger.info("  Running Crew.kickoff_async now …")

        try:
            crew_output: Any = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: asyncio.run(self.crew.kickoff_async(inputs=crew_inputs)),
            )
        except Exception as exc:
            logger.exception("Crew kickoff failed: %s", exc)
            yield f"[Multi-agent RAG error: {exc}]"
            return

        # crew_output.pydantic is the parsed CrewOutput; str() returns the final answer text
        raw = str(crew_output) if crew_output else ""
        logger.info("  CrewOutput raw length: %d chars", len(raw))
        logger.info("  CrewOutput preview : %.200s", raw[:200])

        if not raw or raw == "[NO TASKS OUTPUT]":
            logger.warning("  Crew produced no output — yielding fallback message.")
            yield "[No answer could be generated from the knowledge base. Please try a different question or upload more documents.]"
            return

        # Emit citations metadata first (same format as standard RAG) so frontend can render clickable citations
        if _last_retrieval_records:
            # Build citations metadata with 1-based indexing to match standard RAG format
            citations_metadata = {}
            for idx, record in enumerate(_last_retrieval_records, start=1):
                citations_metadata[idx] = record

            # Normalize scores with min-max scaling (same approach as standard RAG)
            scores = [r.get("score", 0.0) for r in _last_retrieval_records]
            if scores:
                min_score = min(scores)
                max_score = max(scores)
                score_range = max_score - min_score
                for idx in citations_metadata:
                    raw_score = citations_metadata[idx]["score"]
                    if score_range > 0:
                        normalized_score = (raw_score - min_score) / score_range
                    else:
                        normalized_score = 1.0 if raw_score > 0 else 0.0
                    citations_metadata[idx]["score"] = normalized_score

            citations_json = json.dumps({"citations": citations_metadata})
            yield "data: " + citations_json + "\n\n"

        # Stream in small chunks so the frontend sees tokens progressively
        for i in range(0, len(raw), 400):
            yield raw[i:i + 400]