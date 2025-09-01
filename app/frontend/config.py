"""
Configuration constants for LLM WebUI frontend.
"""
import os

# Backend API settings
FASTAPI_URL: str = os.getenv("FASTAPI_URL", "http://localhost:8000")
FASTAPI_TIMEOUT: int = 120
CONNECT_TIMEOUT: int = 5
READ_TIMEOUT: int = 600

# Default model parameters
DEFAULT_TEMPERATURE = 0.7
DEFAULT_CHUNK_SIZE = 500
DEFAULT_N_RESULTS = 3

# Supported file types for document upload
SUPPORTED_EXTENSIONS = [
    # Documents
    "pdf", "doc", "docx", "ppt", "pptx", "rtf", "odt", "epub",
    # Spreadsheets
    "xls", "xlsx", "csv", "tsv",
    # Text files
    "txt", "md", "html", "htm", "org", "rst",
    # Images
    "png", "jpg", "jpeg", "bmp", "tiff", "webp", "gif", "heic", "svg",
]

# System prompts
DEFAULT_RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to enhanced multimodal document knowledge. "
    "Use detailed analysis from the uploaded files, extracted text from images, and comprehensive document "
    "content to provide accurate answers. If the answer requires information not in the context, "
    "clearly state that."
)

# Audio recording settings
AUDIO_RECORDING_CONFIG = {
    "text": "Recording...",
    "recording_color": "#e74c3c",
    "neutral_color": "#2ecc71",
    "icon_name": "microphone"
}
