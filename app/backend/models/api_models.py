"""
API request and response models
"""

from typing import Dict, List, Optional

from pydantic import BaseModel


class ChatMessage(BaseModel):
    """A single message in a conversation."""

    role: str  # "user", "assistant", or "system"
    content: str


class ChatRequestEnhanced(BaseModel):
    message: str
    model: Optional[str] = "default"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048  # Maximum tokens for LLM response generation
    top_p: Optional[float] = 0.9  # Nucleus sampling parameter (0.0 to 1.0)
    frequency_penalty: Optional[float] = 0.0  # Penalize frequent tokens (-2.0 to 2.0)
    repetition_penalty: Optional[float] = 1.0  # Penalize repeated tokens (0.0 to 2.0)
    top_k: Optional[int] = 40  # Top-K sampling cutoff (0 disables the limit)
    min_p: Optional[float] = 0.0  # Minimum token probability floor (0.0 to 1.0)
    seed: Optional[int] = None  # Fixed seed for reproducible outputs (None = random)
    stop: Optional[List[str]] = None  # Sequences that halt generation
    serp_api_key: Optional[str] = None  # Optional SerpAPI key override for web search
    backend: Optional[str] = "ollama"
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    use_web_search: Optional[bool] = False
    is_voice_mode: Optional[bool] = False  # True for voice agent, False for text agent
    enable_generative_ui: Optional[bool] = False
    # Conversation memory fields
    session_id: Optional[str] = None  # Session ID for memory tracking
    conversation_history: Optional[List[ChatMessage]] = None  # Previous messages
    conversation_summary: Optional[str] = None  # Summary of older messages


class RAGQueryRequestEnhanced(BaseModel):
    query: str
    messages: Optional[List[dict]] = []
    model: Optional[str] = "default"
    system_prompt: Optional[str] = "You are a helpful assistant."
    n_results: Optional[int] = 3
    use_multi_agent: Optional[bool] = False
    use_hybrid_search: Optional[bool] = False
    backend: Optional[str] = "ollama"
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    temperature: Optional[float] = 0.7  # Temperature for LLM response generation
    max_tokens: Optional[int] = 2048  # Maximum tokens for LLM response generation
    top_p: Optional[float] = 0.9  # Nucleus sampling parameter (0.0 to 1.0)
    frequency_penalty: Optional[float] = 0.0  # Penalize frequent tokens (-2.0 to 2.0)
    repetition_penalty: Optional[float] = 1.0  # Penalize repeated tokens (0.0 to 2.0)
    min_p: Optional[float] = 0.0  # Minimum token probability floor (0.0 to 1.0)
    seed: Optional[int] = None  # Fixed seed for reproducible outputs (None = random)
    stop: Optional[List[str]] = None  # Sequences that halt generation
    serp_api_key: Optional[str] = None  # Optional SerpAPI key override for web search
    is_voice_mode: Optional[bool] = False  # True for voice agent, False for text agent
    use_web_search: Optional[bool] = False  # Combine retrieved docs with web search results
    # Conversation memory fields
    session_id: Optional[str] = None  # Session ID for memory tracking
    conversation_summary: Optional[str] = None  # Summary of older messages


class ChatResponse(BaseModel):
    response: str
    model: str


class RAGConfigRequest(BaseModel):
    provider: str
    embedding_model: str
    embedding_backend: Optional[str] = None
    embedding_api_base: Optional[str] = None
    embedding_api_key: Optional[str] = None
    chunk_size: int


class STTRequest(BaseModel):
    audio_data: str  # Base64 encoded
    sample_rate: Optional[int] = 16000


class HealthResponse(BaseModel):
    status: str
    checks: Optional[Dict] = None
    timestamp: Optional[float] = None
    error: Optional[str] = None


class ModelListResponse(BaseModel):
    models: List[str]


class OllamaModelInfo(BaseModel):
    """Extended info for a single Ollama model."""

    name: str
    parameter_size: Optional[str] = None
    quantization: Optional[str] = None
    family: Optional[str] = None
    is_local: bool = True
    description: Optional[str] = None


class EnhancedModelListResponse(BaseModel):
    """Response with local + cloud model information."""
    local_models: List[OllamaModelInfo]
    cloud_models: List[OllamaModelInfo]


class RAGStatusResponse(BaseModel):
    status: str
    uploaded_files: Optional[int] = None
    indexed_chunks: Optional[int] = None
    bm25_corpus_size: Optional[int] = None
    message: Optional[str] = None


class RagFileInfo(BaseModel):
    """A single uploaded knowledge-base file."""

    filename: str
    size: int
    indexed: bool = False


class RagFilesResponse(BaseModel):
    """Response for listing uploaded knowledge-base files."""

    files: List[RagFileInfo] = []


class DuplicateCheckRequest(BaseModel):
    """Body for the duplicate-check preflight before upload."""

    filename: str


class DuplicateCheckResponse(BaseModel):
    """Result of a duplicate-check preflight."""

    is_duplicate: bool
    reason: str = ""


class UploadResult(BaseModel):
    filename: str
    status: str
    message: str
    chunks_created: int


class UploadResponse(BaseModel):
    status: str
    summary: Dict
    results: List[UploadResult]


class ResetResponse(BaseModel):
    status: str
    message: str
    files_removed: int


class TranscriptionResponse(BaseModel):
    status: str
    transcription: str
    length: int


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    path: Optional[str] = None


class WebSearchRequest(BaseModel):
    query: str
    num_results: Optional[int] = 10


class WebSearchResult(BaseModel):
    title: str
    link: str
    snippet: str
    position: int


class WebSearchResponse(BaseModel):
    query: str
    results: List[WebSearchResult]
    total_results: int


class SerpTokenStatus(BaseModel):
    status: str
    message: str


class SerpTokenCheck(BaseModel):
    """Optional body for the SerpAPI status endpoint.

    When ``serp_api_key`` is supplied the endpoint validates that specific key
    (e.g. a user-provided client key) instead of the server's ``SERP_API_KEY``
    environment variable.
    """

    serp_api_key: Optional[str] = None


class MemoryStatsResponse(BaseModel):
    """Response model for conversation memory statistics."""

    session_id: str
    message_count: int
    total_tokens: int
    summary_tokens: int
    has_summary: bool
    max_context_tokens: int
    usage_percentage: float
