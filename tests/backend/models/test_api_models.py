"""Unit tests for the backend API pydantic models.

All tests are offline and exercise construction, round-trip equality,
optional-field behaviour, and required-field validation.
"""

import pydantic
import pytest

from app.backend.models.api_models import (
    ChatMessage,
    ChatRequestEnhanced,
    ChatResponse,
    DuplicateCheckRequest,
    DuplicateCheckResponse,
    MemoryStatsResponse,
    RAGConfigRequest,
    RAGQueryRequestEnhanced,
    STTRequest,
    WebSearchRequest,
)


# ── ChatMessage ────────────────────────────────────────────────────
def test_chat_message_roundtrip():
    m = ChatMessage(role="user", content="hi")
    assert m.role == "user" and m.content == "hi"
    dumped = m.model_dump()
    assert ChatMessage(**dumped) == m


def test_chat_message_requires_role_and_content():
    with pytest.raises(pydantic.ValidationError):
        ChatMessage(role="user")
    with pytest.raises(pydantic.ValidationError):
        ChatMessage(content="hi")
    with pytest.raises(pydantic.ValidationError):
        ChatMessage()


# ── ChatRequestEnhanced ────────────────────────────────────────────
def test_chat_request_enhanced_defaults():
    req = ChatRequestEnhanced(message="hello")
    assert req.message == "hello"
    # Spot-check that documented optional fields carry their defaults.
    assert req.model == "default"
    assert req.temperature == 0.7
    assert req.max_tokens == 2048
    assert req.top_p == 0.9
    assert req.frequency_penalty == 0.0
    assert req.repetition_penalty == 1.0
    assert req.top_k == 40
    assert req.min_p == 0.0
    assert req.seed is None
    assert req.stop is None
    assert req.serp_api_key is None
    assert req.backend == "ollama"
    assert req.api_base is None
    assert req.api_key is None
    assert req.use_web_search is False
    assert req.is_voice_mode is False
    assert req.enable_generative_ui is False
    assert req.session_id is None
    assert req.conversation_history is None
    assert req.conversation_summary is None


def test_chat_request_enhanced_requires_message():
    with pytest.raises(pydantic.ValidationError):
        ChatRequestEnhanced(model="gpt")


def test_chat_request_enhanced_optional_overrides():
    req = ChatRequestEnhanced(
        message="hi",
        model="llama3",
        temperature=0.2,
        session_id="abc",
        use_web_search=True,
        conversation_history=[{"role": "user", "content": "x"}],
    )
    assert req.model == "llama3"
    assert req.temperature == 0.2
    assert req.session_id == "abc"
    assert req.use_web_search is True
    assert req.conversation_history[0].role == "user"
    dumped = req.model_dump()
    assert ChatRequestEnhanced(**dumped) == req


# ── RAGQueryRequestEnhanced ────────────────────────────────────────
def test_rag_query_request_enhanced_defaults():
    req = RAGQueryRequestEnhanced(query="what?")
    assert req.query == "what?"
    assert req.model == "default"
    assert req.system_prompt == "You are a helpful assistant."
    assert req.n_results == 3
    assert req.use_multi_agent is False
    assert req.use_hybrid_search is False
    assert req.backend == "ollama"
    assert req.temperature == 0.7
    assert req.max_tokens == 2048


def test_rag_query_request_enhanced_requires_query():
    with pytest.raises(pydantic.ValidationError):
        RAGQueryRequestEnhanced(model="x")


def test_rag_query_request_enhanced_roundtrip():
    req = RAGQueryRequestEnhanced(
        query="q",
        n_results=5,
        use_hybrid_search=True,
        messages=[{"role": "user", "content": "hi"}],
    )
    dumped = req.model_dump()
    assert RAGQueryRequestEnhanced(**dumped) == req


# ── RAGConfigRequest ───────────────────────────────────────────────
def test_rag_config_request_required_fields():
    # provider, embedding_model, and chunk_size are all required (no defaults).
    cfg = RAGConfigRequest(
        provider="ollama",
        embedding_model="nomic-embed-text",
        chunk_size=512,
    )
    assert cfg.provider == "ollama"
    assert cfg.embedding_model == "nomic-embed-text"
    assert cfg.chunk_size == 512
    # Optional embedding backend fields default to None.
    assert cfg.embedding_backend is None
    assert cfg.embedding_api_base is None
    assert cfg.embedding_api_key is None


def test_rag_config_request_optional_overrides():
    cfg = RAGConfigRequest(
        provider="openrouter",
        embedding_model="text-embedding-3-small",
        chunk_size=256,
        embedding_backend="openrouter",
        embedding_api_key="sk-123",
    )
    assert cfg.embedding_backend == "openrouter"
    assert cfg.embedding_api_key == "sk-123"
    dumped = cfg.model_dump()
    assert RAGConfigRequest(**dumped) == cfg


def test_rag_config_request_requires_provider_embedding_model_chunk_size():
    # Missing provider
    with pytest.raises(pydantic.ValidationError):
        RAGConfigRequest(embedding_model="m", chunk_size=1)
    # Missing embedding_model
    with pytest.raises(pydantic.ValidationError):
        RAGConfigRequest(provider="ollama", chunk_size=1)
    # Missing chunk_size
    with pytest.raises(pydantic.ValidationError):
        RAGConfigRequest(provider="ollama", embedding_model="m")


def test_rag_config_request_chunk_size_has_no_range_constraint():
    # The model does NOT constrain chunk_size, so a negative value is accepted
    # at the pydantic layer (out-of-range validation is done elsewhere).
    cfg = RAGConfigRequest(provider="ollama", embedding_model="m", chunk_size=-1)
    assert cfg.chunk_size == -1


# ── STTRequest ─────────────────────────────────────────────────────
def test_stt_request_defaults_and_required():
    req = STTRequest(audio_data="BASE64DATA")
    assert req.audio_data == "BASE64DATA"
    assert req.sample_rate == 16000
    dumped = req.model_dump()
    assert STTRequest(**dumped) == req


def test_stt_request_requires_audio_data():
    with pytest.raises(pydantic.ValidationError):
        STTRequest()
    with pytest.raises(pydantic.ValidationError):
        STTRequest(sample_rate=8000)


def test_stt_request_override_sample_rate():
    req = STTRequest(audio_data="data", sample_rate=8000)
    assert req.sample_rate == 8000


# ── WebSearchRequest ───────────────────────────────────────────────
def test_web_search_request_fields():
    wsr = WebSearchRequest(query="q", num_results=5)
    assert wsr.query == "q"
    assert wsr.num_results == 5


def test_web_search_request_defaults():
    wsr = WebSearchRequest(query="q")
    assert wsr.num_results == 10


def test_web_search_request_requires_query():
    with pytest.raises(pydantic.ValidationError):
        WebSearchRequest(num_results=5)


# ── DuplicateCheckRequest / Response ───────────────────────────────
def test_duplicate_check_request():
    d = DuplicateCheckRequest(filename="a.pdf")
    assert d.filename == "a.pdf"
    with pytest.raises(pydantic.ValidationError):
        DuplicateCheckRequest()
    dumped = d.model_dump()
    assert DuplicateCheckRequest(**dumped) == d


def test_duplicate_check_response_default_reason():
    r = DuplicateCheckResponse(is_duplicate=True)
    assert r.is_duplicate is True
    assert r.reason == ""
    r2 = DuplicateCheckResponse(is_duplicate=False, reason="new file")
    assert r2.reason == "new file"


# ── ChatResponse ───────────────────────────────────────────────────
def test_chat_response_roundtrip():
    r = ChatResponse(response="hello", model="gemma3:1b")
    assert r.response == "hello"
    assert r.model == "gemma3:1b"
    with pytest.raises(pydantic.ValidationError):
        ChatResponse(response="hi")  # missing model
    dumped = r.model_dump()
    assert ChatResponse(**dumped) == r


# ── MemoryStatsResponse (fully-required response model) ────────────
def test_memory_stats_response_requires_all_fields():
    stats = MemoryStatsResponse(
        session_id="s1",
        message_count=10,
        total_tokens=1500,
        summary_tokens=200,
        has_summary=True,
        max_context_tokens=24000,
        usage_percentage=6.25,
    )
    assert stats.session_id == "s1"
    assert stats.usage_percentage == 6.25
    dumped = stats.model_dump()
    assert MemoryStatsResponse(**dumped) == stats


def test_memory_stats_response_missing_field_raises():
    with pytest.raises(pydantic.ValidationError):
        MemoryStatsResponse(
            session_id="s1",
            message_count=10,
            total_tokens=1500,
            summary_tokens=200,
            has_summary=True,
            max_context_tokens=24000,
            # usage_percentage omitted
        )
