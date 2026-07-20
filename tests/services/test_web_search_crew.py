"""Unit tests for ``app.services.web_search_crew`` (offline).

``WebSearchOrchestrator`` wraps ``WebSearchService`` (SerpAPI) and a configured
LLM (via ``llm_gateway.stream_completion``). It is NOT a CrewAI orchestrator, so
we mock the two real external boundaries:

* ``WebSearchService.search`` / ``.format_results`` (SerpAPI) — patched on the
  orchestrator instance so no HTTP occurs,
* ``app.services.web_search_crew.stream_completion`` — patched at module level
  so no real LLM call occurs.

We cover query optimization, result formatting/context/citation building,
synthesis streaming, and the fallback-to-base-LLM paths.
"""
from unittest.mock import AsyncMock, MagicMock, patch

from app.services import web_search_crew as wsc
from app.services.web_search_crew import WebSearchOrchestrator


def _fake_stream(chunks):
    """Return a fresh async generator yielding ``chunks`` each call."""
    async def _gen():
        for c in chunks:
            yield c
    return _gen()


def _make_orchestrator():
    return WebSearchOrchestrator(backend="ollama", model="llama3")


# ── Construction ─────────────────────────────────────────────────────────────
def test_orchestrator_init():
    orch = _make_orchestrator()
    assert orch.backend == "ollama"
    assert orch.requested_model == "llama3"
    assert isinstance(orch.web_search_service, wsc.WebSearchService)
    assert orch.llm_config.backend == "ollama"


# ── Pure helpers ─────────────────────────────────────────────────────────────
def test_build_search_context():
    orch = _make_orchestrator()
    results = [{"position": 1, "title": "T", "link": "http://t", "snippet": "s"}]
    ctx = orch._build_search_context(results)
    assert "T" in ctx and "http://t" in ctx and "s" in ctx
    assert orch._build_search_context([]) == "No search results available."


def test_build_citation_block():
    orch = _make_orchestrator()
    results = [
        {
            "position": 1,
            "title": "Example",
            "link": "https://www.example.com/page",
            "snippet": "A great site",
        }
    ]
    block = orch._build_citation_block(results)
    assert "[1] **Example**" in block
    assert "(example.com)" in block  # www. stripped
    assert "(https://www.example.com/page)" in block
    assert "A great site" in block


def test_build_citation_block_skips_empty_link():
    orch = _make_orchestrator()
    results = [{"position": 1, "title": "NoLink", "link": "", "snippet": "x"}]
    assert orch._build_citation_block(results) == ""


def test_build_citation_block_truncates_long_snippet():
    orch = _make_orchestrator()
    long = "x" * 300
    results = [{"position": 1, "title": "T", "link": "https://example.com", "snippet": long}]
    block = orch._build_citation_block(results)
    assert "..." in block
    assert len(block) < 400


def test_build_conversation_context():
    orch = _make_orchestrator()
    assert orch._build_conversation_context([]) == ""
    hist = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    ctx = orch._build_conversation_context(hist)
    assert "User: hi" in ctx and "Assistant: hello" in ctx


def test_build_conversation_context_limits_recent():
    orch = _make_orchestrator()
    hist = [{"role": "user", "content": f"m{i}"} for i in range(10)]
    ctx = orch._build_conversation_context(hist)
    assert "m0" not in ctx  # older than the last 5 are dropped
    assert "m9" in ctx


def test_create_synthesis_prompt_contains_query():
    orch = _make_orchestrator()
    prompt = orch._create_synthesis_prompt(
        query="What is X?",
        optimized_query="X definition",
        search_context="ctx",
        conversation_context="",
    )
    assert "What is X?" in prompt
    assert "ctx" in prompt


# ── Query optimizer (LLM mocked) ──────────────────────────────────────────────
async def test_query_optimizer_returns_optimized():
    orch = _make_orchestrator()
    with patch.object(
        wsc,
        "stream_completion",
        side_effect=lambda *a, **k: _fake_stream(["best python books 2024"]),
    ):
        optimized = await orch._query_optimizer_process("where can I find good python books")
    assert optimized == "best python books 2024"


async def test_query_optimizer_falls_back_to_original():
    orch = _make_orchestrator()
    with patch.object(
        wsc,
        "stream_completion",
        side_effect=lambda *a, **k: _fake_stream([""]),
    ):
        optimized = await orch._query_optimizer_process("original question here")
    assert optimized == "original question here"


# ── Full workflow (LLM + search mocked) ───────────────────────────────────────
async def test_process_query_streams_synthesis():
    orch = _make_orchestrator()
    orch.web_search_service.search = AsyncMock(
        return_value={"organic_results": [{"title": "T", "link": "http://t", "snippet": "s"}]}
    )
    orch.web_search_service.format_results = MagicMock(
        return_value=[{"position": 1, "title": "T", "link": "http://t", "snippet": "s"}]
    )

    chunks = []
    with patch.object(
        wsc,
        "stream_completion",
        side_effect=lambda *a, **k: _fake_stream(["The answer is 42"]),
    ):
        async for chunk in orch.process_query("question"):
            chunks.append(chunk)

    text = "".join(chunks)
    assert "The answer is 42" in text
    # The orchestrator appends a Sources block after synthesis.
    assert "Sources:" in text
    assert "[1]" in text


async def test_process_query_falls_back_when_no_results():
    orch = _make_orchestrator()
    orch.web_search_service.search = AsyncMock(return_value={"organic_results": []})
    orch.web_search_service.format_results = MagicMock(return_value=[])

    chunks = []
    with patch.object(
        wsc,
        "stream_completion",
        side_effect=lambda *a, **k: _fake_stream(["Fallback answer"]),
    ):
        async for chunk in orch.process_query("question"):
            chunks.append(chunk)

    text = "".join(chunks)
    assert "Fallback answer" in text
    # No synthesis happened, so no Sources block.
    assert "Sources:" not in text


async def test_process_query_falls_back_on_search_error():
    orch = _make_orchestrator()
    orch.web_search_service.search = AsyncMock(side_effect=ValueError("bad key"))

    chunks = []
    with patch.object(
        wsc,
        "stream_completion",
        side_effect=lambda *a, **k: _fake_stream(["Fallback"]),
    ):
        async for chunk in orch.process_query("question"):
            chunks.append(chunk)

    assert "Fallback" in "".join(chunks)
