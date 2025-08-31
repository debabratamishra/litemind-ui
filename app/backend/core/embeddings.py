"""
Core embedding functionality
"""
import logging
from typing import List, Union, Dict

from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)


class LocalHFEmbeddingFunction:
    """Local HuggingFace embedding function with batching"""

    def __init__(self, model_name: str, device: str = None, batch_size: int = 64):
        if device is None:
            device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        logger.info(f"Initialized embedding model: {model_name} on {device}")

    def __call__(self, input: Union[List[str], Dict, str]) -> List[List[float]]:
        """Generate embeddings for input texts"""
        texts = self._extract_texts(input)
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def _extract_texts(self, input: Union[List[str], Dict, str]) -> List[str]:
        """Extract text strings from various input formats"""
        if isinstance(input, dict):
            texts = input.get("input") or input.get("texts") or input.get("documents") or []
        elif isinstance(input, list):
            texts = input
        else:
            texts = [str(input)]
        
        return [str(text) for text in texts]


def create_embedding_function(provider: str, model_name: str, ollama_url: str = None):
    """Factory function to create embedding functions"""
    provider = provider.lower().strip()
    
    if provider == "ollama":
        if not ollama_url:
            raise ValueError("Ollama URL required for Ollama provider")
        return OllamaEmbeddingFunction(
            model_name=model_name,
            url=f"{ollama_url}/api/embeddings",
        )
    elif provider == "huggingface":
        return LocalHFEmbeddingFunction(model_name=model_name)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
