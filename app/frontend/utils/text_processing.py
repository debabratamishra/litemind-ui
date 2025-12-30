"""
Text processing and formatting utilities.
"""
import json
import re
from typing import Tuple, Set


def _load_common_words() -> Set[str]:
    manual = {
        'temperatures', 'information', 'conditions', 'performance', 'requirements',
        'available', 'different', 'important', 'following', 'including',
        'experience', 'development', 'government', 'environment', 'management',
        'international', 'community', 'university', 'organization', 'relationship',
        'opportunity', 'significant', 'particularly', 'understanding', 'responsibilities',
    }

    try:
        from wordfreq import top_n_list

        words = set(w.lower() for w in top_n_list("en", 20000))
        long_words = {w for w in words if len(w) >= 6}
        return long_words.union(manual)
    except Exception:
        return manual


COMMON_WORDS = _load_common_words()


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


def normalize_plain_text_spacing(text: str) -> str:
    """Normalize whitespace for plain text responses without forcing markdown."""
    if not isinstance(text, str) or not text:
        return "" if text is None else str(text)

    normalized = text.replace('\r', '')
    normalized = re.sub(r'[ \t]+\n', '\n', normalized)
    normalized = re.sub(r'\n[ \t]+', '\n', normalized)
    normalized = re.sub(r'\n{3,}', '\n\n', normalized)
    normalized = re.sub(r'[ \t]{2,}', ' ', normalized)

    # Tidy spacing around punctuation and digits
    normalized = re.sub(r'\s+([,.;:!?%])', r'\1', normalized)
    normalized = re.sub(r'\(\s+', '(', normalized)
    normalized = re.sub(r'\s+\)', ')', normalized)
    normalized = re.sub(r'\[\s+', '[', normalized)
    normalized = re.sub(r'\s+\]', ']', normalized)
    normalized = re.sub(r'(?<=\d)\s+(?=\d)', '', normalized)
    normalized = re.sub(r'(?<=\d)\s+(?=%)', '', normalized)
    normalized = re.sub(r'(?<=\w)\s*-\s*(?=\w)', '-', normalized)
    normalized = re.sub(r"\s+'", "'", normalized)
    normalized = re.sub(r"'\s+(?=[a-z])", "'", normalized)

    return normalized.strip()


def format_web_search_response(text: str) -> str:
    """Format web search response text for proper display.
    
    This function handles:
    - Fixing token spacing issues from streaming
    - Formatting source citations properly
    - Converting URLs to clickable markdown links
    - Cleaning up concatenated text issues
    """
    if not isinstance(text, str) or not text:
        return "" if text is None else str(text)
    
    # First, normalize basic spacing
    text = text.replace('\r', '')
    
    text = re.sub(r'(?<=[A-Za-z])\n(?=[A-Za-z](?:\n|[^A-Za-z]))', '', text)
    # Also fix single chars separated by newlines in sequence
    while re.search(r'([A-Za-z])\n([A-Za-z])\n', text):
        text = re.sub(r'([A-Za-z])\n([A-Za-z])\n', r'\1\2\n', text)
    text = re.sub(r'([A-Za-z])\n([A-Za-z])(?=[^a-zA-Z\n]|$)', r'\1\2', text)
    
    def fix_spaced_letters(match):
        """Join spaced single letters into a word."""
        return match.group(0).replace(' ', '')
    
    def fix_word_then_spaced(match):
        word = match.group(1)
        spaced = match.group(2).replace(' ', '')
        return word + spaced
    
    text = re.sub(r'(\w{2,})\s+([a-zA-Z](?:\s+[a-zA-Z]){3,})\b', fix_word_then_spaced, text)
    
    text = re.sub(r'\b([a-zA-Z])((?:\s[a-zA-Z]){3,})\b', fix_spaced_letters, text)
    text = re.sub(r'\b([A-Z]{1,2})\s+([A-Z])\b', r'\1\2', text)
    
    # Fix apostrophe spacing: "here' s" -> "here's", "don' t" -> "don't"
    text = re.sub(r"(\w+)'\s+([a-z])", r"\1'\2", text)
    
    # Fix degree/unit spacing: "27 ° C" -> "27°C", "100 % " -> "100%"
    text = re.sub(r'(\d+)\s*°\s*([CFcf])', r'\1°\2', text)
    text = re.sub(r'(\d+)\s*°', r'\1°', text)
    text = re.sub(r'([A-Z][a-z]+)\s+([a-z]{3,})', lambda m: m.group(1) + m.group(2) if m.group(1).lower() + m.group(2) in COMMON_WORDS else m.group(0), text)
    
    text = re.sub(r'\bS\s+south', 'south', text, flags=re.IGNORECASE)
    text = re.sub(r'\bN\s+north', 'north', text, flags=re.IGNORECASE)
    text = re.sub(r'\bE\s+east', 'east', text, flags=re.IGNORECASE)
    text = re.sub(r'\bW\s+west', 'west', text, flags=re.IGNORECASE)
    
    text = re.sub(r'([A-Z]{2,}):\s*([A-Z])\s+([A-Z])\s*([A-Z])?', 
                  lambda m: f"{m.group(1)}:{m.group(2)}{m.group(3)}{m.group(4) or ''}", text)
    
    for _ in range(5):
        text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
    
    text = re.sub(r'\s+([,.;:!?%])', r'\1', text)
    text = re.sub(r'\$\s+', '$', text)
    text = re.sub(r'Rs\s+', 'Rs ', text)
    
    # Fix decimal numbers with spaces: "178. 88" -> "178.88"
    text = re.sub(r'(\d+)\.\s+(\d+)', r'\1.\2', text)
    
    # Fix numbers with commas and spaces: "1, 000" -> "1,000"
    text = re.sub(r'(\d+),\s+(\d+)', r'\1,\2', text)
    
    # Fix time formatting: "4: 00" -> "4:00"
    text = re.sub(r'(\d+):\s+(\d+)', r'\1:\2', text)
    
    # Fix percentage spacing: "- 2. 17 %" -> "-2.17%"
    text = re.sub(r'-\s*(\d)', r'-\1', text)
    text = re.sub(r'(\d)\s*%', r'\1%', text)
    
    text = re.sub(r'(\d+)-\s+(\w)', r'\1-\2', text)
    text = re.sub(r'(\w)-\s+(\w)', r'\1-\2', text)
    
    text = re.sub(r'\+\s+(\d)', r'+\1', text)
    
    # Fix spacing around brackets for citations
    text = re.sub(r'\[\s*(\d+)\s*\]', r'[\1]', text)
    
    # Fix double spaces
    text = re.sub(r'[ \t]{2,}', ' ', text)
    
    # Fix newline issues
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Format the Sources section
    text = _format_sources_section(text)
    
    return text.strip()


def _format_sources_section(text: str) -> str:
    """Format the Sources section with proper markdown links and structure."""
    # Find the Sources section - handle both with and without newline after "Sources:"
    sources_match = re.search(r'(Sources?:\s*)', text, re.IGNORECASE)
    if not sources_match:
        return text
    
    sources_start = sources_match.start()
    before_sources = text[:sources_start]
    sources_section = text[sources_start:]
    
    # Clean up stray ** markers before the Sources section
    before_sources = re.sub(r'\n\*\*\s*$', '\n', before_sources)
    before_sources = before_sources.rstrip() + '\n'
    
    sources_section = re.sub(
        r'\s*-\s*\[Link\]\([^)]*\)\s*\(\s*\([^)]+\)\s*-\s*\[Link\]\([^)]*\)\s*-\s*Link\s*',
        ' - Link ',
        sources_section
    )
    # Clean simpler duplicates: "[Link]( (domain.com) - [Link]( (domain.com) - Link"
    sources_section = re.sub(
        r'\[Link\]\(\s*\([^)]+\)\s*-\s*\[Link\]\(\s*\([^)]+\)\s*-\s*Link',
        'Link',
        sources_section
    )
    # Remove malformed [Link]( patterns without proper URLs
    sources_section = re.sub(r'\[Link\]\(\s*\(', '(', sources_section)
    # Clean up "- [Link]( (domain) -" patterns
    sources_section = re.sub(r'-\s*\[Link\]\(\s*\(([^)]+)\)\s*-', r'(\1) -', sources_section)
    
    # Fix malformed URLs with spaces
    sources_section = re.sub(
        r'(https?)\s*:\s*/\s*/\s*',
        r'\1://',
        sources_section
    )
    sources_section = re.sub(r'\s*\.\s*(?=com|org|net|edu|gov|io|co)', '.', sources_section)
    sources_section = re.sub(r'(?<=\w)\.\s+(?=com|org|net|edu|gov|io|co)', '.', sources_section)
    
    # Split sources that are all on one line: "[1] ... [2] ..." -> "[1] ...\n[2] ..."
    # Add newline before each [N] citation except the first one after "Sources:"
    sources_section = re.sub(r'\s+\[(\d+)\]', r'\n[\1]', sources_section)
    
    # Ensure "Sources:" is followed by newline
    sources_section = re.sub(r'^(Sources?:)\s*', r'\1\n', sources_section, flags=re.IGNORECASE)
    
    # Format each source line properly
    lines = sources_section.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue  # Skip empty lines, we'll add proper spacing
        
        # Check if this is the "Sources:" header (with or without ** markers)
        if re.match(r'^\*?\*?\s*Sources?:?\s*\*?\*?$', line, re.IGNORECASE):
            formatted_lines.append('\n**Sources:**\n')
            continue
        
        # Skip standalone ** markers (artifact from bad formatting)
        if line == '**':
            continue
        
        # Check if this is a source line starting with [N]
        source_match = re.match(r'^\[(\d+)\]\s*(.+)$', line)
        if source_match:
            formatted_lines.append(_format_single_source(line))
        else:
            formatted_lines.append(line)
    
    return before_sources + '\n'.join(formatted_lines)


def _format_single_source(source_line: str) -> str:
    """Format a single source citation line.
    
    Format:
    [N] Title (domain) - Link
        Description text here...
    
    (blank line before next citation)
    """
    # Extract the citation number
    match = re.match(r'^\[(\d+)\]\s*(.+)$', source_line)
    if not match:
        return source_line
    
    num = match.group(1)
    rest = match.group(2)
    
    # Clean up any duplicate [Link] patterns first
    rest = re.sub(r'\s*-\s*\[Link\]\(\s*\([^)]+\)\s*-\s*\[Link\]\(\s*\([^)]+\)\s*-\s*Link\s*', ' - Link ', rest)
    rest = re.sub(r'\[Link\]\(\s*\(', '(', rest)
    rest = re.sub(r'\)\s*-\s*\[Link\]\(\s*\(', ') - (', rest)
    rest = re.sub(r'\[Link\]\(\s*$', '', rest)
    
    # Fix any single-character-per-line issues in rest
    rest = re.sub(r'(?<=[A-Za-z])\n(?=[A-Za-z])', '', rest)
    while re.search(r'([A-Za-z])\n([A-Za-z])\n', rest):
        rest = re.sub(r'([A-Za-z])\n([A-Za-z])\n', r'\1\2\n', rest)
    rest = re.sub(r'([A-Za-z])\n([A-Za-z])(?=[^a-zA-Z\n]|$)', r'\1\2', rest)
    
    # Check if already has a proper markdown link [Link](url) - skip reformatting
    if re.search(r'\[Link\]\(https?://[^)]+\)', rest):
        return f"\n[{num}] {rest}\n"
    
    # Try to extract URL from the line
    url_match = re.search(r'(https?://[^\s<>"\[\]]+)', rest)
    
    if url_match:
        url = url_match.group(1).rstrip('.,;:!?)')
        url = re.sub(r'\s+', '', url)
        
        title_part = rest[:url_match.start()].strip()
        title_part = re.sub(r'[—–\-]+\s*$', '', title_part).strip()
        title_part = re.sub(r'\([^)]+\)\s*$', '', title_part).strip()
        
        domain_match = re.match(r'https?://(?:www\.)?([^/]+)', url)
        domain = domain_match.group(1) if domain_match else ''
        
        after_url = rest[url_match.end():].strip()
        after_url = re.sub(r'^[—–\-]+\s*', '', after_url).strip()
        
        if title_part:
            if after_url:
                return f"\n[{num}] **{title_part}** ({domain}) - [Link]({url})\n    *{after_url}*\n"
            return f"\n[{num}] **{title_part}** ({domain}) - [Link]({url})\n"
        else:
            return f"\n[{num}] [{domain}]({url})\n"
    
    # Handle format without full URL: "Title (domain.com) - Link Description"
    domain_patterns = re.findall(r'\(([a-zA-Z][a-zA-Z0-9\-\.]*\.(com|org|net|edu|gov|io|co|in|uk|de|fr|jp|au|ca|info|biz))\)', rest)
    if domain_patterns:
        domain = domain_patterns[0][0]
        
        domain_pos = rest.find(f'({domain})')
        if domain_pos > 0:
            title_part = rest[:domain_pos].strip()
            title_part = re.sub(r'\s*\([A-Z]+\.[A-Z]+\)\s*', ' ', title_part).strip()
            title_part = re.sub(r'[—–\-]+\s*$', '', title_part).strip()
            title_part = re.sub(r'^\*\*(.+)\*\*$', r'\1', title_part)
            
            after_domain = rest[domain_pos + len(f'({domain})'):].strip()
            after_domain = re.sub(r'^[—–\-]+\s*\[?Link\]?\(?\s*', '', after_domain, flags=re.IGNORECASE).strip()
            after_domain = re.sub(r'^\([^)]+\)\s*-\s*\[?Link\]?\(?\s*', '', after_domain).strip()
            
            if title_part:
                if after_domain:
                    # Put description on new line, italicized and indented
                    return f"\n[{num}] **{title_part}** ({domain}) - Link\n    *{after_domain}*\n"
                return f"\n[{num}] **{title_part}** ({domain})\n"
            else:
                return f"\n[{num}] {domain}\n"
    
    # Fallback: just return the line as-is but on a new line
    return f"\n[{num}] {rest}\n"


def fix_streaming_token_spacing(accumulated: str, new_chunk: str) -> str:
    """Intelligently join streaming tokens to avoid spacing issues.
    
    This function handles the common issue where streaming LLM responses
    produce individual tokens that get joined with incorrect spacing.
    """
    if not accumulated:
        return new_chunk
    
    if not new_chunk:
        return accumulated
    
    # Characters that should NOT have a space before them
    no_space_before = set('.,;:!?%)]}\'"\n')
    # Characters that should NOT have a space after them
    no_space_after = set('([{$\n+-')
    
    last_char = accumulated[-1] if accumulated else ''
    first_char = new_chunk[0] if new_chunk else ''
    
    # Determine if we need a space
    needs_space = True
    
    # Don't add space if accumulated ends with space or newline
    if last_char in ' \n\t':
        needs_space = False
    # Don't add space if new chunk starts with space or newline
    elif first_char in ' \n\t':
        needs_space = False
    # Don't add space before punctuation
    elif first_char in no_space_before:
        needs_space = False
    # Don't add space after certain characters
    elif last_char in no_space_after:
        needs_space = False
    # Don't add space between digits (for numbers like 178.88)
    elif last_char.isdigit() and first_char in '.,:%':
        needs_space = False
    elif last_char in '.,' and first_char.isdigit():
        needs_space = False
    # Don't add space for contractions
    elif first_char == "'":
        needs_space = False
    # Don't add space after opening brackets
    elif last_char in '([{':
        needs_space = False
    # Don't add space before closing brackets
    elif first_char in ')]}':
        needs_space = False
    # Don't add space before hyphen for negative numbers or hyphenated words
    elif first_char == '-' and (accumulated.rstrip()[-1:].isalpha() or accumulated.rstrip()[-1:].isdigit()):
        needs_space = False
    # Don't add space between hyphen and following character
    elif last_char == '-':
        needs_space = False
    
    if needs_space:
        return accumulated + ' ' + new_chunk
    return accumulated + new_chunk
