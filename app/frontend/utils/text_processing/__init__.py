import asyncio
from typing import Tuple

from .markdown import clean_markdown_text, clean_text_formatting
from .spacing import fix_streaming_token_spacing, normalize_plain_text_spacing
from .thinking import extract_thinking_content
from .unescape import unescape_text
from .url import sanitize_links
from .web_search import format_web_search_response

# --- Asynchronous wrappers (Thread-Safe Execution) ---

async def clean_text_formatting_async(text: str) -> str:
    """Async wrapper for clean_text_formatting."""
    return await asyncio.to_thread(clean_text_formatting, text)


async def clean_markdown_text_async(text: str) -> str:
    """Async wrapper for clean_markdown_text."""
    return await asyncio.to_thread(clean_markdown_text, text)


async def format_web_search_response_async(text: str) -> str:
    """Async wrapper for format_web_search_response."""
    return await asyncio.to_thread(format_web_search_response, text)


async def sanitize_links_async(text: str) -> str:
    """Async wrapper for sanitize_links."""
    return await asyncio.to_thread(sanitize_links, text)


async def extract_thinking_content_async(text: str) -> Tuple[str, str]:
    """Async wrapper for extract_thinking_content."""
    return await asyncio.to_thread(extract_thinking_content, text)


async def normalize_plain_text_spacing_async(text: str) -> str:
    """Async wrapper for normalize_plain_text_spacing."""
    return await asyncio.to_thread(normalize_plain_text_spacing, text)
