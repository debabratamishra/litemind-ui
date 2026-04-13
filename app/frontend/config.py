"""
Configuration constants for LLM WebUI frontend.
"""
import os

from app.core.rag_formats import SUPPORTED_EXTENSIONS as RAG_SUPPORTED_EXTENSIONS

# Backend API settings
FASTAPI_URL: str = os.getenv("FASTAPI_URL", "http://localhost:8000")
FASTAPI_TIMEOUT: int = 120
CONNECT_TIMEOUT: int = 5
READ_TIMEOUT: int = 600

# Default model parameters
DEFAULT_TEMPERATURE = 0.7
DEFAULT_CHUNK_SIZE = 500
DEFAULT_N_RESULTS = 3
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TOP_P = 0.9
DEFAULT_FREQUENCY_PENALTY = 0.0
DEFAULT_REPETITION_PENALTY = 1.0

# Supported file types for document upload
SUPPORTED_EXTENSIONS = list(RAG_SUPPORTED_EXTENSIONS)

# System prompts - Text Mode (detailed, comprehensive responses)
DEFAULT_RAG_SYSTEM_PROMPT = (
    "You are a helpful Retrieval Augmented Generation assistant optimized for a small language model with access to enhanced multimodal document knowledge. "
    "Answer strictly and only from the provided context (uploaded documents, extracted text from images, and any supplied snippets). "
    "Do not use outside knowledge or guess; if the context is insufficient, say so clearly and briefly. "
    "Be concise, factual, and grounded in the provided context. If multiple passages apply, synthesize them naturally. Do not mention citations, source numbers, or filenames unless the user explicitly asks for sources. "
    "If the user asks for something not covered in the context, respond that the information is not in the provided documents."
)

DEFAULT_CHAT_SYSTEM_PROMPT_TEXT = (
    "You are a knowledgeable and helpful AI assistant. Provide detailed, accurate, and well-structured responses. "
    "When answering complex questions, break down your explanation into clear sections. "
    "If you need more information to answer accurately, ask specific clarifying questions. "
    "Be thorough and comprehensive in your responses."
)

DEFAULT_RAG_SYSTEM_PROMPT_TEXT = DEFAULT_RAG_SYSTEM_PROMPT

# System prompts - Voice Mode
DEFAULT_CHAT_SYSTEM_PROMPT_VOICE = (
    "You are a friendly AI assistant speaking to a person. "
    "Answer in natural speech, 1-2 short sentences max. "
    "Do NOT use bullet points, numbered lists, markdown, or URLs unless the user explicitly asks for a link. "
    "Answer based on what you know, and if the context doesn't have the answer, say so briefly. "
    "Keep it conversational and light; avoid jargon. "
    "If something is unclear, ask follow up brief clarifying questions."
)

DEFAULT_RAG_SYSTEM_PROMPT_VOICE = (
    "You are a helpful assistant with access to document knowledge. "
    "Speak like a person: 1-2 short sentences max, no bullet points, no markdown, no links unless the user asks for them. "
    "If the context lacks the answer, say so briefly and ask follow up brief clarifying questions. "
    "Do not mention source numbers, citations, or filenames unless the user explicitly asks for sources. "
    "Answer based on what you know, and if the context doesn't have the answer, say so briefly. "
    "Keep it natural and concise."
)

# Audio recording settings
AUDIO_RECORDING_CONFIG = {
    "text": "Recording...",
    "recording_color": "#e74c3c",
    "neutral_color": "#2ecc71",
    "icon_name": "microphone"
}
