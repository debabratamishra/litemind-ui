"""Unit tests for ``app.services.rag_service`` (offline).

The RAG service wraps ChromaDB (vector store), a BM25 keyword index, embedding
providers, and the LLM gateway. All external boundaries are mocked:

* ``chromadb.PersistentClient`` / the shared client — patched so NO real
  on-disk ChromaDB is created,
* the embedding function — set to a ``MagicMock`` so no model download occurs,
* ``stream_completion`` (LLM gateway) — patched so no network/LLM call happens,
* ``asyncio.to_thread`` ingestion helpers — replaced with tiny fakes.

Temp dirs are bound to the relevant config paths via the shared
``tmp_chroma_dir`` / ``tmp_upload_dir`` fixtures.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.services import rag_service as rs


# ── Fixture ──────────────────────────────────────────────────────────────────
@pytest.fixture
def rag_service(tmp_chroma_dir, tmp_upload_dir):
    """A ``RAGService`` wired to fully mocked ChromaDB + embedding boundaries."""
    collection = MagicMock(name="text_collection")
    # Empty collection so index-rebuild paths short-circuit safely.
    collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}
    client = MagicMock(name="chroma_client")
    client.get_collection.return_value = collection
    client.get_or_create_collection.return_value = collection

    with patch.object(rs.RAGService, "_get_chroma_db_path", return_value=str(tmp_chroma_dir)), patch.object(
        rs.RAGService, "_get_upload_directory", return_value=str(tmp_upload_dir)
    ), patch.object(rs.RAGService, "_get_or_create_client", return_value=client):
        svc = rs.RAGService()
        svc.client = client
        svc.text_collection = collection
        # Provide a mock embedding function so no model is loaded.
        svc.embedding_function = MagicMock(return_value=[[0.1, 0.2, 0.3]])
        yield svc


# ── Module-level helpers ──────────────────────────────────────────────────────
def test_module_importable_true():
    assert rs._module_importable("os") is True


def test_module_importable_false_for_missing():
    assert rs._module_importable("this_module_does_not_exist_abc123") is False


def test_tokenize_text_splits_words():
    toks = rs._tokenize_text("The quick brown fox")
    assert "quick" in toks
    assert "brown" in toks
    assert all(isinstance(t, str) for t in toks)


def test_tokenize_text_empty():
    assert rs._tokenize_text("") == []


def test_flatten_metadata_recurses():
    flat = rs._flatten_metadata({"a": 1, "b": {"c": 2, "d": {"e": 3}}})
    assert flat["a"] == 1
    # Nested keys are joined with underscores (not dots) in the real impl.
    assert flat["b_c"] == 2
    assert flat["b_d_e"] == 3


def test_flatten_metadata_lists_json_encoded():
    flat = rs._flatten_metadata({"tags": ["x", "y"]})
    assert flat["tags"] == '["x", "y"]'


def test_flatten_metadata_scalars_and_none():
    flat = rs._flatten_metadata({"s": "v", "i": 1, "f": 1.5, "b": True, "n": None})
    assert flat["s"] == "v"
    assert flat["i"] == 1
    assert flat["n"] is None


def test_flatten_metadata_non_dict():
    assert rs._flatten_metadata("not a dict") == {}


# ── Text preprocessing / cleaning ─────────────────────────────────────────────
def test_preprocess_text_removes_stopwords_and_punct():
    svc = rs.RAGService.__new__(rs.RAGService)
    svc.stop_words = rs._load_stop_words()
    toks = svc.preprocess_text("The cat sat on the mat.")
    assert "cat" in toks
    assert "the" not in toks
    assert "." not in toks


def test_preprocess_text_empty():
    svc = rs.RAGService.__new__(rs.RAGService)
    svc.stop_words = rs._load_stop_words()
    assert svc.preprocess_text("   ") == []


def test_clean_text_for_indexing():
    svc = rs.RAGService.__new__(rs.RAGService)
    out = svc._clean_text_for_indexing("Page 1 HEADERS:\n\n  Hello   world  ")
    assert "Page 1 HEADERS:" not in out
    assert "Hello world" in out


# ── Chunking ──────────────────────────────────────────────────────────────────
def test_chunk_text_empty():
    svc = rs.RAGService.__new__(rs.RAGService)
    assert svc.chunk_text("") == []
    assert svc.chunk_text("   \n  ") == []


def test_chunk_text_paragraph_aware():
    svc = rs.RAGService.__new__(rs.RAGService)
    text = "First paragraph about cats.\n\nSecond paragraph about dogs."
    # With a small chunk size the two paragraphs become separate chunks.
    chunks = svc.chunk_text(text, chunk_size=20)
    assert len(chunks) == 2
    assert "First paragraph" in chunks[0]
    assert "Second paragraph" in chunks[1]


def test_chunk_text_overlap_within_bounds():
    svc = rs.RAGService.__new__(rs.RAGService)
    # Long single paragraph forces sentence splitting + overlap.
    text = (
        "Sentence one describes the alpha module. "
        + "Sentence two describes the beta module. " * 20
    )
    chunk_size = 200
    chunks = svc.chunk_text(text, chunk_size=chunk_size)
    assert len(chunks) >= 2
    # Overlap head can be a full prior chunk, so a chunk is bounded by ~2*chunk_size.
    for c in chunks:
        assert len(c) <= 2 * chunk_size + 2


def test_fallback_character_chunking():
    svc = rs.RAGService.__new__(rs.RAGService)
    text = "a" * 250
    chunks = svc._fallback_character_chunking(text, chunk_size=100, overlap=20)
    assert all(len(c) <= 100 for c in chunks)
    assert len(chunks) >= 2


# ── BM25 search ───────────────────────────────────────────────────────────────
def _seed_bm25(svc, chunks, scores):
    svc.stop_words = rs._load_stop_words()
    svc.document_chunks = chunks
    svc.chunk_ids = [f"id{i}" for i in range(len(chunks))]
    svc.chunk_metadata_by_id = {
        cid: {"filename": f"f{i}"} for i, cid in enumerate(svc.chunk_ids)
    }
    model = MagicMock(name="bm25")
    model.get_scores.return_value = np.array(scores, dtype=float)
    svc.bm25_model = model


def test_bm25_search_ranks_and_excludes_zero():
    svc = rs.RAGService.__new__(rs.RAGService)
    _seed_bm25(svc, ["alpha beta", "gamma delta", "epsilon"], [3.0, 1.0, 0.0])
    res = svc.bm25_search("anything", n_results=3)
    assert [r[0] for r in res] == ["id0", "id1"]  # id2 score 0 excluded
    assert all(r[2] > 0 for r in res)


def test_bm25_search_empty_when_no_model():
    svc = rs.RAGService.__new__(rs.RAGService)
    svc.bm25_model = None
    svc.document_chunks = []
    assert svc.bm25_search("q") == []


def test_bm25_search_empty_when_query_tokenizes_to_nothing():
    svc = rs.RAGService.__new__(rs.RAGService)
    _seed_bm25(svc, ["a b", "c d"], [1.0, 1.0])
    svc.stop_words = rs._load_stop_words()
    # "the" is a stopword and the only token -> no query tokens.
    assert svc.bm25_search("the") == []


def test_bm25_search_records_preserves_metadata():
    svc = rs.RAGService.__new__(rs.RAGService)
    _seed_bm25(svc, ["alpha", "beta"], [2.0, 1.0])
    records = svc.bm25_search_records("alpha topic", n_results=2)
    assert records[0]["id"] == "id0"
    assert records[0]["retrieval_method"] == "bm25"
    assert records[0]["metadata"]["filename"] == "f0"


# ── Vector (semantic) search ───────────────────────────────────────────────────
def test_vector_search_text_with_embedding_function(rag_service):
    rag_service.text_collection.query.return_value = {
        "ids": [["id1"]],
        "documents": [["doc text"]],
        "distances": [[0.2]],
    }
    out = rag_service.vector_search_text("question", n_results=1)
    assert out == [("id1", "doc text", 0.8)]
    # Query used our own embedding (query_embeddings), not raw text.
    _, kwargs = rag_service.text_collection.query.call_args
    assert "query_embeddings" in kwargs


def test_vector_search_text_without_embedding_function(rag_service):
    rag_service.embedding_function = None
    rag_service.text_collection.query.return_value = {
        "ids": [["id1"]],
        "documents": [["doc text"]],
        "distances": [[0.1]],
    }
    out = rag_service.vector_search_text("question", n_results=1)
    assert out[0][2] == 0.9
    _, kwargs = rag_service.text_collection.query.call_args
    assert "query_texts" in kwargs


def test_vector_search_text_empty_results(rag_service):
    rag_service.text_collection.query.return_value = {"documents": [[]]}
    assert rag_service.vector_search_text("q") == []


def test_vector_search_text_handles_error(rag_service):
    rag_service.text_collection.query.side_effect = RuntimeError("chroma down")
    assert rag_service.vector_search_text("q") == []


def test_vector_search_records_with_metadata(rag_service):
    rag_service.text_collection.query.return_value = {
        "ids": [["id1"]],
        "documents": [["doc text"]],
        "metadatas": [[{"filename": "f"}]],
        "distances": [[0.3]],
    }
    records = rag_service.vector_search_records("q", n_results=1)
    assert records[0]["id"] == "id1"
    assert records[0]["retrieval_method"] == "semantic"
    assert records[0]["metadata"] == {"filename": "f"}


def test_vector_search_records_handles_error(rag_service):
    rag_service.text_collection.query.side_effect = RuntimeError("boom")
    assert rag_service.vector_search_records("q") == []


# ── Reciprocal Rank Fusion ──────────────────────────────────────────────────────
def test_reciprocal_rank_fusion_merges_and_orders():
    svc = rs.RAGService.__new__(rs.RAGService)
    bm25 = [{"id": "a"}, {"id": "b"}]
    vec = [{"id": "b"}, {"id": "c"}]
    fused = svc.reciprocal_rank_fusion(bm25, vec, k=60)
    # "b" appears in both -> highest score, must be first.
    assert fused[0] == "b"
    assert set(fused) == {"a", "b", "c"}


def test_reciprocal_rank_fusion_accepts_tuple_ids():
    svc = rs.RAGService.__new__(rs.RAGService)
    fused = svc.reciprocal_rank_fusion([("a",)], [("b",)], k=60)
    assert set(fused) == {"a", "b"}


# ── Hybrid search ────────────────────────────────────────────────────────────
def test_hybrid_search_records(rag_service):
    # BM25 path
    model = MagicMock(name="bm25")
    model.get_scores.return_value = np.array([2.0, 1.0], dtype=float)
    rag_service.bm25_model = model
    rag_service.document_chunks = ["alpha text", "beta text"]
    rag_service.chunk_ids = ["bid1", "bid2"]
    rag_service.chunk_metadata_by_id = {"bid1": {"filename": "f1"}, "bid2": {"filename": "f2"}}
    # Vector path
    rag_service.text_collection.query.return_value = {
        "ids": [["vid1"]],
        "documents": [["vec doc"]],
        "metadatas": [[{"filename": "f3"}]],
        "distances": [[0.2]],
    }
    records = rag_service.hybrid_search_records("alpha topic", n_results=5)
    ids = {r["id"] for r in records}
    assert "bid1" in ids and "vid1" in ids


def test_hybrid_search_returns_contents(rag_service):
    rag_service.bm25_model = None  # fall back to vector only
    rag_service.text_collection.query.return_value = {
        "ids": [["vid1"]],
        "documents": [["vec doc"]],
        "metadatas": [[{}]],
        "distances": [[0.2]],
    }
    assert rag_service.hybrid_search("q") == ["vec doc"]


def test_hybrid_search_records_empty_when_no_results(rag_service):
    rag_service.bm25_model = None
    rag_service.text_collection.query.return_value = {"documents": [[]]}
    assert rag_service.hybrid_search_records("q") == []


# ── Retrieval query building ───────────────────────────────────────────────────
def test_build_retrieval_query_no_history():
    svc = rs.RAGService.__new__(rs.RAGService)
    assert svc.build_retrieval_query("hello") == "hello"


def test_build_retrieval_query_with_history():
    svc = rs.RAGService.__new__(rs.RAGService)
    out = svc.build_retrieval_query(
        "hello", messages=[{"role": "system", "content": "ignore"}, {"role": "user", "content": "prev"}]
    )
    assert "prev" in out and "hello" in out and "ignore" not in out


def test_get_retrieval_records_hybrid_vs_semantic(rag_service):
    # No bm25 model -> semantic
    rag_service.bm25_model = None
    rag_service.text_collection.query.return_value = {
        "ids": [["x"]],
        "documents": [["d"]],
        "metadatas": [[{}]],
        "distances": [[0.1]],
    }
    recs = rag_service.get_retrieval_records("q", n_results=1, use_hybrid_search=True)
    assert recs[0]["id"] == "x"


# ── Prompt composition ─────────────────────────────────────────────────────────
def test_build_grounded_user_prompt_no_records():
    svc = rs.RAGService.__new__(rs.RAGService)
    prompt = svc.build_grounded_user_prompt("Q?", [])
    assert "No relevant context" in prompt
    assert "Q?" in prompt


def test_build_grounded_user_prompt_with_records():
    svc = rs.RAGService.__new__(rs.RAGService)
    records = [{"content": "Some fact about cats", "metadata": {"filename": "f"}}]
    prompt = svc.build_grounded_user_prompt("Tell me about cats?", records)
    assert "[1]" in prompt
    assert "Some fact about cats" in prompt


def test_build_grounded_user_prompt_voice_mode():
    svc = rs.RAGService.__new__(rs.RAGService)
    prompt = svc.build_grounded_user_prompt("Q?", [], voice_mode=True)
    assert "briefly" in prompt.lower()


def test_build_cited_user_prompt_alias():
    svc = rs.RAGService.__new__(rs.RAGService)
    records = [{"content": "x", "metadata": {}}]
    assert svc.build_cited_user_prompt("Q?", records) == svc.build_grounded_user_prompt("Q?", records)


def test_truncate_source_content():
    svc = rs.RAGService.__new__(rs.RAGService)
    long = "word " * 1000
    out = svc._truncate_source_content(long, max_chars=100)
    assert len(out) <= 100


# ── Source label / section-title helpers ───────────────────────────────────────
def test_is_meaningful_section_title_filters_boilerplate():
    svc = rs.RAGService.__new__(rs.RAGService)
    assert svc._is_meaningful_section_title("Table 1", {}) is False
    assert svc._is_meaningful_section_title("Introduction", {}) is True


def test_format_source_label_with_section_and_page():
    svc = rs.RAGService.__new__(rs.RAGService)
    label = svc._format_source_label({"filename": "doc.pdf", "page_number": 3, "section_title": "Intro"}, 0)
    assert "doc.pdf" in label
    assert "page 3" in label
    assert "Intro" in label


# ── Indexing ───────────────────────────────────────────────────────────────────
def test_index_chunk_batch_with_embeddings(rag_service):
    rag_service.text_collection.add.reset_mock()
    batch = [{"content": "hello world", "metadata": {"filename": "f"}}]
    rag_service._index_chunk_batch(batch, "doc1")
    args, kwargs = rag_service.text_collection.add.call_args
    assert kwargs.get("embeddings") is not None
    assert "hello world" in kwargs["documents"]
    assert len(rag_service.bm25_corpus) == 1


def test_index_chunk_batch_without_embeddings(rag_service):
    rag_service.embedding_function = None
    rag_service.text_collection.add.reset_mock()
    batch = [{"content": "plain text", "metadata": {}}]
    rag_service._index_chunk_batch(batch, "doc2")
    _, kwargs = rag_service.text_collection.add.call_args
    assert "embeddings" not in kwargs
    assert kwargs["documents"] == ["plain text"]


def test_index_chunk_batch_empty_returns_early(rag_service):
    rag_service.text_collection.add.reset_mock()
    rag_service._index_chunk_batch([], "doc3")
    rag_service.text_collection.add.assert_not_called()


def test_index_chunk_batch_raises_on_error(rag_service):
    rag_service.text_collection.add.side_effect = RuntimeError("index fail")
    with pytest.raises(RuntimeError):
        rag_service._index_chunk_batch([{"content": "x", "metadata": {}}], "doc4")


# ── Document ingestion (add_document) ───────────────────────────────────────────
async def test_add_document_success(tmp_path, rag_service):
    f = tmp_path / "doc.txt"
    f.write_text("some content")
    rag_service._calculate_file_hash = MagicMock(return_value="hash123")
    rag_service._ingest_file_with_enhancement = MagicMock(
        return_value=([{"content": "real content here", "metadata": {"content_type": "text_content"}}], [], [])
    )
    result = await rag_service.add_document(str(f), "doc.txt")
    assert result["status"] == "success"
    assert result["chunks_created"] == 1
    rag_service.text_collection.add.assert_called()


async def test_add_document_duplicate_skipped(tmp_path, rag_service):
    f = tmp_path / "doc.txt"
    f.write_text("content")
    rag_service._calculate_file_hash = MagicMock(return_value="hash123")
    rag_service.processed_files = {"doc.txt": {"hash": "h", "chunk_count": 5}}
    result = await rag_service.add_document(str(f), "doc.txt")
    assert result["status"] == "duplicate"


async def test_add_document_missing_file(rag_service):
    rag_service._calculate_file_hash = MagicMock(return_value="hash123")
    result = await rag_service.add_document("/no/such/file.txt", "file.txt")
    assert result["status"] == "error"


# ── Lifecycle operations ────────────────────────────────────────────────────────
def test_recreate_collection(rag_service):
    async def _run():
        await rag_service.recreate_collection()

    import asyncio

    asyncio.run(_run())
    rag_service.client.delete_collection.assert_called_with(name="documents_text")
    rag_service.client.create_collection.assert_called()


def test_reset_system(rag_service):
    rag_service.processed_files = {"f": {}}

    import asyncio

    async def _run():
        await rag_service.reset_system()

    asyncio.run(_run())
    rag_service.client.delete_collection.assert_called()
    rag_service.client.create_collection.assert_called()
    assert rag_service.processed_files == {}


def test_remove_processed_file(rag_service):
    rag_service.processed_files = {"f.pdf": {"hash": "h", "chunk_count": 2}}
    rag_service.file_hashes = {"h": "f.pdf"}
    rag_service.text_collection.get.return_value = {"ids": ["id1", "id2"]}
    assert rag_service.remove_processed_file("f.pdf") is True
    rag_service.text_collection.delete.assert_called_with(ids=["id1", "id2"])


def test_remove_processed_file_missing(rag_service):
    assert rag_service.remove_processed_file("nope.pdf") is False


def test_get_capabilities(rag_service):
    with patch.object(rs, "get_ingestion_capabilities", return_value={"local_pipeline": {}}):
        caps = rag_service.get_capabilities()
    assert caps["status"] == "ready"
    assert "supported_extensions" in caps


# ── Query (async generator) ──────────────────────────────────────────────────────
async def test_query_yields_citations_then_stream(rag_service):
    rag_service.text_collection.query.return_value = {
        "ids": [["id1"]],
        "documents": [["context passage"]],
        "metadatas": [[{"filename": "f"}]],
        "distances": [[0.2]],
    }

    async def fake_stream(*args, **kwargs):
        yield "token1"
        yield "token2"

    with patch.object(rs, "stream_completion", side_effect=fake_stream):
        chunks = [c async for c in rag_service.query("question?")]

    assert chunks
    assert chunks[0].startswith("data:")
    assert "citations" in chunks[0]
    assert "token1" in chunks[1]
    assert "token2" in chunks[2]


async def test_query_hybrid_path(rag_service):
    model = MagicMock(name="bm25")
    model.get_scores.return_value = np.array([1.0], dtype=float)
    rag_service.bm25_model = model
    rag_service.document_chunks = ["bm25 doc"]
    rag_service.chunk_ids = ["bid"]
    rag_service.chunk_metadata_by_id = {"bid": {"filename": "f"}}
    rag_service.text_collection.query.return_value = {
        "ids": [["vid"]],
        "documents": [["vec doc"]],
        "metadatas": [[{}]],
        "distances": [[0.1]],
    }

    async def fake_stream(*args, **kwargs):
        yield "ans"

    with patch.object(rs, "stream_completion", side_effect=fake_stream):
        chunks = [c async for c in rag_service.query("q", use_hybrid_search=True)]

    assert chunks[0].startswith("data:")
    assert "ans" in chunks[-1]


async def test_query_empty_results_still_emits_citations(rag_service):
    rag_service.bm25_model = None
    rag_service.text_collection.query.return_value = {"documents": [[]]}

    async def fake_stream(*args, **kwargs):
        yield "noop"

    with patch.object(rs, "stream_completion", side_effect=fake_stream):
        chunks = [c async for c in rag_service.query("q")]

    assert chunks[0].startswith("data:")
    assert "noop" in chunks[1]
