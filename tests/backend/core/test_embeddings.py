"""Tests for ``app.backend.core.embeddings`` (offline).

No model is downloaded and no network call is made:

* ``LocalHFEmbeddingFunction`` is replaced with a lightweight fake for the
  HuggingFace path (its real ``__init__`` loads a ``sentence-transformers`` model).
* The Ollama path constructs a ``chromadb`` ``OllamaEmbeddingFunction`` whose
  ``__init__`` only builds an ``httpx.Client`` (no request) -- safe offline.
* The LiteLLM path (OpenRouter / Nvidia NIM) only builds a resolved config object
  in ``__init__`` and never touches the network unless ``__call__`` is invoked.
"""
import pytest

from app.backend.core import embeddings


def test_resolve_embedding_provider_unknown_raises():
    with pytest.raises(ValueError):
        embeddings.resolve_embedding_provider("not-a-provider")


def test_resolve_embedding_provider_known():
    assert embeddings.resolve_embedding_provider("huggingface") == "huggingface"
    assert embeddings.resolve_embedding_provider("OLLAMA") == "ollama"
    assert (
        embeddings.resolve_embedding_provider("litellm", embedding_backend="openrouter")
        == "openrouter"
    )
    # litellm without a supported backend is rejected.
    with pytest.raises(ValueError):
        embeddings.resolve_embedding_provider("litellm", embedding_backend="bogus")


def test_create_embedding_function_huggingface_returns_callable(monkeypatch):
    class _FakeHF:
        def __init__(self, model_name: str, device=None, batch_size: int = 64):
            self.model_name = model_name

        def __call__(self, input):
            return [[0.0, 0.0]]

    monkeypatch.setattr(embeddings, "LocalHFEmbeddingFunction", _FakeHF)

    fn = embeddings.create_embedding_function(provider="huggingface", model_name="x")
    assert callable(fn)
    assert isinstance(fn, _FakeHF)


def test_create_embedding_function_ollama_returns_callable():
    fn = embeddings.create_embedding_function(
        provider="ollama",
        model_name="nomic-embed-text",
        ollama_url="http://localhost:11434",
    )
    assert callable(fn)
    assert isinstance(fn, embeddings.OllamaEmbeddingFunction)


def test_create_embedding_function_ollama_requires_url():
    with pytest.raises(ValueError):
        embeddings.create_embedding_function(provider="ollama", model_name="x")


def test_create_embedding_function_openrouter_returns_callable():
    fn = embeddings.create_embedding_function(
        provider="openrouter",
        model_name="text-embedding-3-small",
        api_base="https://openrouter.ai/api/v1",
        api_key="test-key",
    )
    assert callable(fn)
    assert isinstance(fn, embeddings.LiteLLMEmbeddingFunction)
    assert fn.config.backend == "openrouter"


def test_create_embedding_function_nvidia_nim_returns_callable():
    fn = embeddings.create_embedding_function(
        provider="nvidia_nim",
        model_name="nv-embed-qa",
        api_base="https://api.nvcf.nim.ai/v1",
        api_key="test-key",
    )
    assert callable(fn)
    assert isinstance(fn, embeddings.LiteLLMEmbeddingFunction)
    assert fn.config.backend == "nvidia_nim"
