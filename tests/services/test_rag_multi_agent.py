"""Unit tests for ``app.services.rag_multi_agent`` (offline).

The CrewAI multi-agent RAG orchestrator is wrapped so the heavy external
boundaries are mocked:

* ``_load_crewai_types`` — patched with ``MagicMock`` stand-ins for
  ``Agent`` / ``LLM`` / ``Task`` / ``Crew`` / ``Process`` / ``Tool`` / ``tool``
  so NO real CrewAI objects (which may validate against a live model) are built,
* the retrieve-knowledge tool — invoked against a mocked ChromaDB collection,
* ``crew.kickoff_async`` — patched so NO real crew runs.

When CrewAI is actually importable we also exercise the real ``tool`` decorator
and ``_load_crewai_types`` to confirm the public surface is intact.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services import rag_multi_agent as rma


def _reset_records():
    rma._last_retrieval_records = []


# ── Availability / type loading ─────────────────────────────────────────────────
def test_multi_agent_rag_available_shape():
    out = rma.multi_agent_rag_available()
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert isinstance(out[0], bool)
    assert isinstance(out[1], str)


def test_load_crewai_types_returns_seven():
    # Only meaningful when CrewAI is installed; otherwise expect ImportError.
    try:
        types = rma._load_crewai_types()
    except ImportError:
        pytest.skip("crewai not installed")
    assert len(types) == 7


# ── Retrieve tool (real @tool decorator when available) ─────────────────────────
def _make_tool(collection):
    """Build the retrieve_knowledge tool against a mocked collection."""
    try:
        from crewai.tools.base_tool import tool
    except ImportError:
        pytest.skip("crewai not installed")
    rag_service = MagicMock()
    rag_service.text_collection = collection
    return rma._build_retrieve_tool(rag_service, tool)


def test_retrieve_tool_empty_collection():
    collection = MagicMock()
    collection.count.return_value = 0
    tool = _make_tool(collection)
    out = tool.run("any question")
    assert out == "[No results]"


def test_retrieve_tool_returns_formatted_chunks():
    collection = MagicMock()
    collection.count.return_value = 2
    collection.query.return_value = {
        "documents": [["first passage", "second passage"]],
        "metadatas": [[{"source": "doc.pdf", "page_number": 4}, {"source": "doc.pdf"}]],
        "distances": [[0.1, 0.3]],
    }
    tool = _make_tool(collection)
    out = tool.run("question about topic")
    assert "first passage" in out
    assert "second passage" in out
    assert "Relevance:" in out
    # Captured for citation metadata.
    assert len(rma._last_retrieval_records) == 2


def test_retrieve_tool_query_error():
    collection = MagicMock()
    collection.count.return_value = 5
    collection.query.side_effect = RuntimeError("chroma down")
    tool = _make_tool(collection)
    assert tool.run("q") == "[No results]"


def test_retrieve_tool_unavailable_collection():
    rag_service = MagicMock()
    rag_service.text_collection = None  # triggers except path
    try:
        from crewai.tools.base_tool import tool
    except ImportError:
        pytest.skip("crewai not installed")
    built = rma._build_retrieve_tool(rag_service, tool)
    assert built.run("q") == "[No results - knowledge base unavailable]"


# ── Orchestrator: assembly ──────────────────────────────────────────────────────
def _mock_crewai_types():
    Agent, LLM, Task, Crew, Process, Tool, tool = (MagicMock() for _ in range(7))  # noqa: N806
    crew_instance = Crew.return_value
    Agent.return_value = MagicMock(name="agent")
    LLM.return_value = MagicMock(name="llm")
    Task.return_value = MagicMock(name="task")
    return (Agent, LLM, Task, Crew, Process, Tool, tool), crew_instance


def test_orchestrator_assembly():
    _reset_records()
    types, crew_instance = _mock_crewai_types()
    with patch.object(rma, "_load_crewai_types", return_value=types):
        orch = rma.CrewAIRAGOrchestrator(rag_service=MagicMock(), backend="ollama", model_name="gemma3:1b")
    # Crew was constructed with researcher + synthesizer agents and sequential process.
    assert orch.crew is crew_instance
    assert orch.researcher is not None
    assert orch.synthesiser is not None
    assert orch.backend == "ollama"
    # resolve_backend_config normalises the ollama model to the `ollama_chat/` prefix.
    assert orch.model_name == "ollama_chat/gemma3:1b"


def test_orchestrator_assembly_ollama_model_normalised():
    _reset_records()
    types, _ = _mock_crewai_types()
    with patch.object(rma, "_load_crewai_types", return_value=types):
        rma.CrewAIRAGOrchestrator(rag_service=MagicMock(), backend="ollama", model_name="ollama_chat/gemma3:1b")
    # The orchestrator rewrites `ollama_chat/...` -> `ollama/...` for the CrewAI LLM call.
    llm_mock = types[1]
    _, llm_kwargs = llm_mock.call_args
    assert llm_kwargs["model"] == "ollama/gemma3:1b"


# ── Orchestrator: query (routing + result merge) ─────────────────────────────────
async def test_orchestrator_query_streams_answer():
    _reset_records()
    types, crew_instance = _mock_crewai_types()
    orch = None
    with patch.object(rma, "_load_crewai_types", return_value=types):
        orch = rma.CrewAIRAGOrchestrator(rag_service=MagicMock())

    # Simulate a crew that yields a final answer string.
    answer = MagicMock()
    answer.__str__.return_value = "The answer is 42."
    crew_instance.kickoff_async = AsyncMock(return_value=answer)

    chunks = [c async for c in orch.query("What is the answer?", system_prompt="sys")]
    assert any("The answer is 42." in c for c in chunks)
    crew_instance.kickoff_async.assert_awaited()


async def test_orchestrator_query_emits_citations_from_records():
    _reset_records()
    # Seed retrieval records captured by the tool so citations are merged in.
    rma._last_retrieval_records = [
        {
            "id": "doc.pdf",
            "content": "fact",
            "score": 0.8,
            "retrieval_method": "multi-agent retrieval",
            "metadata": {"filename": "doc.pdf", "page_number": 2},
        }
    ]
    types, crew_instance = _mock_crewai_types()
    with patch.object(rma, "_load_crewai_types", return_value=types):
        orch = rma.CrewAIRAGOrchestrator(rag_service=MagicMock())

    answer = MagicMock()
    answer.__str__.return_value = "Cited answer."
    crew_instance.kickoff_async = AsyncMock(return_value=answer)

    chunks = [c async for c in orch.query("q?", system_prompt="sys")]
    assert chunks[0].startswith("data:")
    assert "citations" in chunks[0]
    assert any("Cited answer." in c for c in chunks)


async def test_orchestrator_query_no_output_fallback():
    _reset_records()
    types, crew_instance = _mock_crewai_types()
    with patch.object(rma, "_load_crewai_types", return_value=types):
        orch = rma.CrewAIRAGOrchestrator(rag_service=MagicMock())

    empty = MagicMock()
    empty.__str__.return_value = "[NO TASKS OUTPUT]"
    crew_instance.kickoff_async = AsyncMock(return_value=empty)

    chunks = [c async for c in orch.query("q?", system_prompt="sys")]
    assert any("No answer could be generated" in c for c in chunks)


async def test_orchestrator_query_kickoff_error():
    _reset_records()
    types, crew_instance = _mock_crewai_types()
    with patch.object(rma, "_load_crewai_types", return_value=types):
        orch = rma.CrewAIRAGOrchestrator(rag_service=MagicMock())

    crew_instance.kickoff_async = AsyncMock(side_effect=RuntimeError("crew failed"))

    chunks = [c async for c in orch.query("q?", system_prompt="sys")]
    assert any("Multi-agent RAG error" in c for c in chunks)
