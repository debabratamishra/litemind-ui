from typing import Tuple
from app.core.text_markup import extract_tagged_sections
from .common import _collapse_newlines

def extract_thinking_content(text: str) -> Tuple[str, str]:
    """Extract thinking/reasoning content from text."""
    thinking_content, clean_text = extract_tagged_sections(text, ("think", "thinking", "reasoning", "thought"))
    clean_text = _collapse_newlines(clean_text, 2).strip()
    combined_thinking = '\n\n'.join(thinking_content) if thinking_content else ''

    return combined_thinking, clean_text
