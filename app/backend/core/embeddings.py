"""
Core embedding functionality
"""
import logging
from typing import Any, Dict, List, Union

import litellm
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from app.services.llm_gateway import resolve_embedding_config

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)


def _extract_texts(input: Union[List[str], Dict, str]) -> List[str]:
    """Extract text strings from various input formats."""
    if isinstance(input, dict):
        texts = input.get("input") or input.get("texts") or input.get("documents") or []
    elif isinstance(input, list):
        texts = input
    else:
        texts = [str(input)]

    return [str(text) for text in texts]


class LocalHFEmbeddingFunction:
    """Local HuggingFace embedding function with batching"""

    def __init__(self, model_name: str, device: str = None, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer
        if device is None:
            device = "cuda" if torch and torch.cuda.is_available() else "cpu"

        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        logger.info(f"Initialized embedding model: {model_name} on {device}")

    def __call__(self, input: Union[List[str], Dict, str]) -> List[List[float]]:
        """Generate embeddings for input texts"""
        texts = _extract_texts(input)

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()


class LiteLLMEmbeddingFunction:
    """Embedding function backed by the LiteLLM SDK."""

    def __init__(
        self,
        backend: str,
        model_name: str,
        api_base: str | None = None,
        api_key: str | None = None,
        batch_size: int = 32,
    ):
        self.config = resolve_embedding_config(
            backend=backend,
            model=model_name,
            api_base=api_base,
            api_key=api_key,
        )
        self.batch_size = batch_size
        logger.info(
            "Initialized LiteLLM embedding model: backend=%s model=%s api_base=%s",
            self.config.backend,
            self.config.model,
            self.config.api_base,
        )

    def __call__(self, input: Union[List[str], Dict, str]) -> List[List[float]]:
        texts = _extract_texts(input)
        embeddings: List[List[float]] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = litellm.embedding(
                model=self.config.model,
                input=batch,
                api_base=self.config.api_base,
                api_key=self.config.api_key,
            )
            embeddings.extend(_extract_embedding_vectors(response))

        return embeddings


def _extract_embedding_vectors(response: Any) -> List[List[float]]:
    """Extract vectors from a LiteLLM embedding response."""
    data = getattr(response, "data", None)
    if data is None and isinstance(response, dict):
        data = response.get("data")

    if not isinstance(data, list):
        raise ValueError("Embedding response did not contain a data list.")

    vectors: List[List[float]] = []
    for item in data:
        embedding = getattr(item, "embedding", None)
        if embedding is None and isinstance(item, dict):
            embedding = item.get("embedding")
        if embedding is None:
            raise ValueError("Embedding response item did not include an embedding vector.")
        vectors.append([float(value) for value in embedding])

    return vectors


def resolve_embedding_provider(provider: str, embedding_backend: str | None = None) -> str:
    """Normalize embedding provider names, including legacy LiteLLM configs."""
    resolved_provider = provider.lower().strip()

    if resolved_provider == "litellm":
        resolved_backend = (embedding_backend or "ollama").lower().strip()
        if resolved_backend in {"ollama", "openrouter", "nvidia_nim"}:
            return resolved_backend
        raise ValueError(f"Unsupported LiteLLM embedding backend: {resolved_backend}")

    if resolved_provider in {"ollama", "huggingface", "openrouter", "nvidia_nim"}:
        return resolved_provider

    raise ValueError(f"Unsupported provider: {provider}")


def create_embedding_function(
    provider: str,
    model_name: str,
    ollama_url: str = None,
    *,
    embedding_backend: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
):
    """Factory function to create embedding functions"""
    provider = resolve_embedding_provider(provider, embedding_backend)

    if provider == "ollama":
        if not ollama_url:
            raise ValueError("Ollama URL required for Ollama provider")
        return OllamaEmbeddingFunction(
            model_name=model_name,
            url=f"{ollama_url}/api/embeddings",
        )
    elif provider == "huggingface":
        return LocalHFEmbeddingFunction(model_name=model_name)
    elif provider == "openrouter":
        return LiteLLMEmbeddingFunction(
            backend="openrouter",
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
        )
    elif provider == "nvidia_nim":
        return LiteLLMEmbeddingFunction(
            backend="nvidia_nim",
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")
