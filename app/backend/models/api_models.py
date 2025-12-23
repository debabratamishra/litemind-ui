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
    backend: Optional[str] = "ollama"
    hf_token: Optional[str] = None
    use_web_search: Optional[bool] = False
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
    hf_token: Optional[str] = None
    # Conversation memory fields
    session_id: Optional[str] = None  # Session ID for memory tracking
    conversation_summary: Optional[str] = None  # Summary of older messages


class ChatResponse(BaseModel):
    response: str
    model: str


class RAGConfigRequest(BaseModel):
    provider: str
    embedding_model: str
    chunk_size: int


class STTRequest(BaseModel):
    audio_data: str  # Base64 encoded
    sample_rate: Optional[int] = 16000


class VLLMTokenRequest(BaseModel):
    token: str


class VLLMModelRequest(BaseModel):
    model_name: str
    dtype: Optional[str] = "auto"
    max_model_len: Optional[int] = None
    gpu_memory_utilization: Optional[float] = 0.9


class HealthResponse(BaseModel):
    status: str
    checks: Optional[Dict] = None
    timestamp: Optional[float] = None
    error: Optional[str] = None


class ModelListResponse(BaseModel):
    models: List[str]


class RAGStatusResponse(BaseModel):
    status: str
    uploaded_files: Optional[int] = None
    indexed_chunks: Optional[int] = None
    bm25_corpus_size: Optional[int] = None
    message: Optional[str] = None


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


class VLLMStatusResponse(BaseModel):
    running: bool
    current_model: Optional[str] = None


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


class MemoryStatsResponse(BaseModel):
    """Response model for conversation memory statistics."""
    session_id: str
    message_count: int
    total_tokens: int
    summary_tokens: int
    has_summary: bool
    max_context_tokens: int
    usage_percentage: float
