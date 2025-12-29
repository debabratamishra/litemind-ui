"""
Text processing and formatting utilities.
"""
import json
import re
from typing import Tuple


def unescape_text(s: str) -> str:
    """Convert visible escape sequences into their actual characters."""
    if not isinstance(s, str):
        s = str(s)

    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]

    try:
        s = json.loads(f'"{s}"')
    except Exception:
        s = s.replace('\\n', '\n').replace('\\t', '\t')

    return s.replace('\\r\\n', '\n').replace('\r\n', '\n')


def looks_like_url(text: str) -> bool:
    """Return True if the text appears to be a URL."""
    return bool(re.match(r"^\s*(https?://|www\.)", text, re.IGNORECASE))


def clean_url(url: str) -> str:
    """Normalize a URL string that may contain stray spaces."""
    u = url.strip()
    
    if u.startswith("www."):
        u = "https://" + u
    
    u = re.sub(r"(?i)(https?)\s*:\s*/\s*/", r"\1://", u)
    u = re.sub(r"\s*\.\s*", ".", u)
    u = re.sub(r"\s*/\s*", "/", u)
    u = re.sub(r"\s*:\s*", ":", u)
    u = re.sub(r"(https?):(//)", r"\1://", u)
    
    parts = u.split()
    if len(parts) > 1:
        clean_url_result = parts[0]
        clean_url_result = re.sub(r'\s+', '', clean_url_result)
        return clean_url_result
    else:
        return re.sub(r'\s+', '', u)


def sanitize_links(text: str) -> str:
    """Fix common hyperlink formatting issues in Markdown text."""
    def _md_link_repl(m: re.Match) -> str:
        disp, url = m.group(1), m.group(2)
        clean_url_result = clean_url(url)
        clean_disp = clean_url(disp) if looks_like_url(disp) else disp
        return f"[{clean_disp}]({clean_url_result})"

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _md_link_repl, text)

    def _bare_url_repl(m: re.Match) -> str:
        url_part = m.group(0)
        cleaned = clean_url(url_part)
        return cleaned

    # Fixed ReDoS vulnerability - use more restrictive pattern with length limit
    bare_url_pattern = r"(?i)https?://[a-zA-Z0-9\-._%~:/?#\[\]@!$&'()*+,;=]{1,2000}(?=\s+[A-Z][a-z]|\s*[.!?]\s|$|\s+\w+:)"
    simple_url_pattern = r"(?i)https?\s*:\s*/\s*/[^\s<>\"'\(\)\[\]{}|\\^`]+(?:[^\s<>\"'\(\)\[\]{}|\\^`.,;!?]|[.,;!?](?=\s|$))*"
    
    text = re.sub(bare_url_pattern, _bare_url_repl, text)
    text = re.sub(simple_url_pattern, _bare_url_repl, text)
    return text


def extract_thinking_content(text: str) -> Tuple[str, str]:
    """Extract thinking/reasoning content from text."""
    thinking_patterns = [
        r'<think>(.*?)</think>',
        r'<thinking>(.*?)</thinking>',
        r'<reasoning>(.*?)</reasoning>',
        r'<thought>(.*?)</thought>',
    ]
    
    thinking_content = []
    clean_text = text
    
    for pattern in thinking_patterns:
        matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            thinking_content.append(match.group(1).strip())
            clean_text = re.sub(pattern, '', clean_text, flags=re.DOTALL | re.IGNORECASE)
    
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text).strip()
    combined_thinking = '\n\n'.join(thinking_content) if thinking_content else ''
    
    return combined_thinking, clean_text


def clean_text_formatting(text: str) -> str:
    """Clean up common text formatting issues while preserving word spacing."""
    text = _fix_concatenated_text_after_urls(text)
    text = _fix_numbered_lists(text)
    
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'([a-z])\s*\n\s*(#{1,6}\s+[A-Z])', r'\1\n\n\2', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\s{2,}([.!?,:;])', r' \1', text)
    text = re.sub(r'([.!?])\s*\n\s*([a-z])', r'\1 \2', text)
    text = re.sub(r'(https?://[^\s]+)([A-Z][a-z])', r'\1 \2', text)
    
    return text


def _fix_numbered_lists(text: str) -> str:
    """Fix numbered list formatting by ensuring proper line breaks."""
    numbered_list_pattern = r'([.!?])\s+(\d+\.\s+)([A-Z])'
    text = re.sub(numbered_list_pattern, r'\1\n\n\2\3', text)
    
    colon_list_pattern = r'(:)\s+(\d+\.\s+)([A-Z])'
    text = re.sub(colon_list_pattern, r'\1\n\n\2\3', text)
    
    word_list_pattern = r'([a-z])\s+(\d+\.\s+)([A-Z])'
    text = re.sub(word_list_pattern, r'\1\n\n\2\3', text)
    
    bullet_pattern = r'([.!?])\s+(•\s+)([A-Z])'
    text = re.sub(bullet_pattern, r'\1\n\n\2\3', text)
    
    bullet_colon_pattern = r'(:)\s+(•\s+)([A-Z])'
    text = re.sub(bullet_colon_pattern, r'\1\n\n\2\3', text)
    
    text = re.sub(r'^(\d+\.)\s+', r'\1 ', text, flags=re.MULTILINE)
    text = re.sub(r'^(•)\s+', r'\1 ', text, flags=re.MULTILINE)
    
    return text


def _fix_concatenated_text_after_urls(text: str) -> str:
    """Fix concatenated text that appears after URLs."""
    concatenation_fixes = [
        (r'(https?://[^\s]+/)Thisisthemainwebsiteandthemostcommonwaytoaccessit\.?', 
         r'\1 This is the main website and the most common way to access it.'),
        (r'(https?://[^\s]+)Thisisthemainwebsiteandthemostcommonwaytoaccessit\.?', 
         r'\1 This is the main website and the most common way to access it.'),
        (r'(https?://[^\s]+/)andyou\'?rethere\.?', 
         r'\1 and you\'re there.'),
        (r'(https?://[^\s]+)andyou\'?rethere\.?', 
         r'\1 and you\'re there.'),
        (r'(https?://[^\s]+/)andyourethere\.?', 
         r'\1 and you\'re there.'),
        (r'(https?://[^\s]+)andyourethere\.?', 
         r'\1 and you\'re there.'),
    ]
    
    for pattern, replacement in concatenation_fixes:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text


def clean_markdown_text(text: str) -> str:
    """Clean markdown text for better rendering."""
    if not text.strip():
        return text
    
    text = re.sub(r'^(#{1,6})\s*([^#\n]+)', r'\1 \2', text, flags=re.MULTILINE)
    text = re.sub(r'([^\n])\n(#{1,6}\s)', r'\1\n\n\2', text)
    text = re.sub(r'(#{1,6}[^\n]+)\n([^\n#])', r'\1\n\n\2', text)
    text = re.sub(r'^(\s*[-*+]\s)', r'\1', text, flags=re.MULTILINE)
    text = re.sub(r'^(\s*\d+\.\s)', r'\1', text, flags=re.MULTILINE)
    text = re.sub(r'([^\n])\n(\d+\.\s)', r'\1\n\n\2', text)
    text = re.sub(r'(\d+\.\s[^\n]+)\s+(\d+\.\s)', r'\1\n\n\2', text)
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    
    return text.strip()
