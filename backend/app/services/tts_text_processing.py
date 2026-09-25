from typing import Optional

# Sentence-ending punctuation for streaming chunking
_SENTENCE_END_CHARS = frozenset('.!?;:')


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def _split_on_sentence_endings(text: str) -> list:
    """Split text on sentence-ending punctuation (., !, ?, ;, :) followed by optional whitespace."""
    if not text:
        return []
    parts = []
    start = 0
    for i, ch in enumerate(text):
        if ch in _SENTENCE_END_CHARS:
            parts.append(text[start:i])
            # Skip whitespace after punctuation
            j = i + 1
            while j < len(text) and text[j].isspace():
                j += 1
            start = j
    if start < len(text):
        parts.append(text[start:])
    return parts


# ---------------------------------------------------------------------------
# Code and URL removal
# ---------------------------------------------------------------------------

def _remove_inline_code(text: str) -> str:
    """Remove inline code spans delimited by single backticks."""
    if '`' not in text:
        return text
    result = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == '`':
            j = text.find('`', i + 1)
            if j != -1 and j > i + 1:  # Non-empty content between backticks
                i = j + 1
                continue
        result.append(text[i])
        i += 1
    return ''.join(result)


def _remove_urls(text: str) -> str:
    """Remove http/https/www URLs from text, replacing with ' [link] '."""
    # Characters that terminate a URL (whitespace + special chars from original regex)
    url_terminators = frozenset(' \t\n\r<>"{}|\\^`[]')

    result = []
    i = 0
    n = len(text)
    while i < n:
        if text[i:i+8] == 'https://':
            result.append(' [link] ')
            j = i + 8
            while j < n and text[j] not in url_terminators:
                j += 1
            i = j
        elif text[i:i+7] == 'http://':
            result.append(' [link] ')
            j = i + 7
            while j < n and text[j] not in url_terminators:
                j += 1
            i = j
        elif text[i:i+4] == 'www.':
            result.append(' [link] ')
            j = i + 4
            while j < n and text[j] not in url_terminators:
                j += 1
            i = j
        else:
            result.append(text[i])
            i += 1
    return ''.join(result)


def _remove_file_paths(text: str) -> str:
    """Remove Unix and Windows file paths from text."""
    result = []
    i = 0
    n = len(text)
    while i < n:
        removed = False

        # Unix path: /[\w.-]+ (one or more segments, optional trailing /)
        if text[i] == '/' and i + 1 < n:
            nxt = text[i + 1]
            if nxt.isalnum() or nxt in '._-':
                j = i + 1
                # Scan first segment
                while j < n and (text[j].isalnum() or text[j] in '._-'):
                    j += 1
                # Scan additional segments
                while j < n and text[j] == '/' and j + 1 < n:
                    peek = text[j + 1]
                    if peek.isalnum() or peek in '._-':
                        j += 1  # skip the /
                        while j < n and (text[j].isalnum() or text[j] in '._-'):
                            j += 1
                    else:
                        break
                # Optional trailing /
                if j < n and text[j] == '/':
                    j += 1
                if j > i + 1:
                    i = j
                    removed = True

        # Windows path: [A-Za-z]:\\[\w\\.-]+ (one or more segments)
        if not removed and i + 2 < n:
            if text[i].isalpha() and text[i + 1] == ':' and text[i + 2] == '\\':
                j = i + 3
                # Scan first segment
                while j < n and (text[j].isalnum() or text[j] in '._- '):
                    j += 1
                # Scan additional segments
                while j < n and text[j] == '\\' and j + 1 < n:
                    j += 1
                    while j < n and (text[j].isalnum() or text[j] in '._- '):
                        j += 1
                if j > i + 3:
                    i = j
                    removed = True

        if not removed:
            result.append(text[i])
            i += 1

    return ''.join(result)


# ---------------------------------------------------------------------------
# Brace block removal
# ---------------------------------------------------------------------------

def _remove_brace_blocks(text: str, open_char: str, close_char: str) -> str:
    """Remove brace-delimited blocks that contain no nested braces of the same type."""
    if open_char not in text:
        return text
    result = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == open_char:
            j = i + 1
            while j < n and text[j] != close_char and text[j] != open_char:
                j += 1
            if j < n and text[j] == close_char:
                i = j + 1
                continue
        result.append(text[i])
        i += 1
    return ''.join(result)


# ---------------------------------------------------------------------------
# Markdown formatting removal
# ---------------------------------------------------------------------------

def _remove_delimiter_pair(text: str, delimiter: str, forbid: Optional[str] = None) -> str:
    """Remove paired delimiter markers from text, keeping the inner content.

    Args:
        text: Input text.
        delimiter: The delimiter string to match (e.g. ``**``, ``*``, ``__``, ``_``, ``~~``).
        forbid: If set, the content between delimiters must NOT contain this character.
    """
    if delimiter not in text:
        return text

    result = []
    i = 0
    n = len(text)
    dl_len = len(delimiter)

    while i < n:
        if text[i:i + dl_len] == delimiter:
            # Find closing delimiter
            found = False
            j = i + dl_len
            while j <= n - dl_len:
                if text[j:j + dl_len] == delimiter:
                    inner = text[i + dl_len:j]
                    if inner and (forbid is None or forbid not in inner):
                        result.append(inner)
                        i = j + dl_len
                        found = True
                        break
                j += 1
            if not found:
                result.append(text[i])
                i += 1
        else:
            result.append(text[i])
            i += 1

    return ''.join(result)


def _remove_markdown_formatting(text: str) -> str:
    """Remove bold, italic, and strikethrough markdown formatting from text.

    Processes delimiters in order: ``**`` (bold), ``*`` (italic), ``__`` (bold alt),
    ``_`` (italic alt), ``~~`` (strikethrough). Each pass removes the delimiter pair
    while preserving the inner content.
    """
    text = _remove_delimiter_pair(text, '**', forbid='*')
    text = _remove_delimiter_pair(text, '*', forbid='*')
    text = _remove_delimiter_pair(text, '__', forbid='_')
    text = _remove_delimiter_pair(text, '_', forbid='_')
    text = _remove_delimiter_pair(text, '~~', forbid='~')
    return text


def _remove_markdown_links(text: str) -> str:
    """Remove markdown links ``[text](url)``, keeping the link text."""
    if '[' not in text:
        return text

    result = []
    i = 0
    n = len(text)

    while i < n:
        if text[i] == '[':
            close_bracket = text.find(']', i + 1)
            if close_bracket != -1 and close_bracket + 1 < n and text[close_bracket + 1] == '(':
                close_paren = text.find(')', close_bracket + 2)
                if close_paren != -1:
                    # Keep the link text, skip the entire [text](url) construct
                    result.append(text[i + 1:close_bracket])
                    i = close_paren + 1
                    continue
        result.append(text[i])
        i += 1

    return ''.join(result)


def _remove_markdown_headers(text: str) -> str:
    """Remove markdown header markers (``#``) at start of lines."""
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith('#'):
            # Count leading # characters
            j = 0
            while j < len(stripped) and stripped[j] == '#':
                j += 1
            # Skip the #s and any following space
            if j < len(stripped) and stripped[j] == ' ':
                j += 1
            cleaned.append(stripped[j:])
        else:
            cleaned.append(line)
    return '\n'.join(cleaned)


def _remove_list_markers(text: str) -> str:
    """Remove bullet points and numbered list markers from start of lines."""
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.lstrip()
        indent = line[:len(line) - len(stripped)]

        if not stripped:
            cleaned.append(line)
            continue

        # Bullet markers: - * +
        if stripped[0] in '-*+' and len(stripped) >= 2 and stripped[1].isspace():
            # Skip the bullet and following whitespace
            j = 2
            while j < len(stripped) and stripped[j].isspace():
                j += 1
            cleaned.append(indent + stripped[j:])
        # Numbered list: 1. 2. etc.
        elif stripped[0].isdigit():
            j = 1
            while j < len(stripped) and stripped[j].isdigit():
                j += 1
            if j < len(stripped) and stripped[j] == '.' and j + 1 < len(stripped) and stripped[j + 1].isspace():
                j += 2  # skip the dot and following space
                while j < len(stripped) and stripped[j].isspace():
                    j += 1
                cleaned.append(indent + stripped[j:])
            else:
                cleaned.append(line)
        else:
            cleaned.append(line)

    return '\n'.join(cleaned)


# ---------------------------------------------------------------------------
# HTML tag removal
# ---------------------------------------------------------------------------

def _strip_html_tags(text: str) -> str:
    """Remove HTML tags from text without regex (O(n), no ReDoS)."""
    out = []
    in_tag = False
    for ch in text:
        if ch == '<':
            in_tag = True
        elif ch == '>':
            in_tag = False
        elif not in_tag:
            out.append(ch)
    return ''.join(out)


# ---------------------------------------------------------------------------
# Emoji and special character removal
# ---------------------------------------------------------------------------

def _is_emoji_codepoint(cp: int) -> bool:
    """Check if a Unicode codepoint falls within known emoji and symbol ranges.

    Covers the same ranges as the original regex pattern in _remove_emojis().
    O(1) per character via integer comparisons — no regex engine overhead.
    """
    return (
        (0x1F600 <= cp <= 0x1F64F) or  # emoticons
        (0x1F300 <= cp <= 0x1F5FF) or  # symbols & pictographs
        (0x1F680 <= cp <= 0x1F6FF) or  # transport & map symbols
        (0x1F1E0 <= cp <= 0x1F1FF) or  # flags
        (0x02702 <= cp <= 0x027B0) or  # dingbats
        (0x024C2 <= cp <= 0x1F251) or  # enclosed characters
        (0x1F900 <= cp <= 0x1F9FF) or  # supplemental symbols
        (0x1FA00 <= cp <= 0x1FA6F) or  # chess symbols
        (0x1FA70 <= cp <= 0x1FAFF) or  # symbols and pictographs extended-A
        (0x02600 <= cp <= 0x026FF) or  # misc symbols
        (0x02700 <= cp <= 0x027BF) or  # dingbats
        (0x1F000 <= cp <= 0x1F02F) or  # mahjong tiles
        (0x1F0A0 <= cp <= 0x1F0FF)    # playing cards
    )


def _remove_special_chars(text: str) -> str:
    """Replace special characters that don't sound good when read aloud."""
    _special = frozenset('#@$%^&*()+={}[]|\\<>~')
    result = []
    for ch in text:
        if ch in _special:
            result.append(' ')
        else:
            result.append(ch)
    return ''.join(result)


# ---------------------------------------------------------------------------
# Character compression and whitespace normalization
# ---------------------------------------------------------------------------

def _compress_repeated_chars(text: str, chars: frozenset, replacement: str) -> str:
    """Compress runs of repeated characters into a single replacement string.

    If a run has 2 or more of the same character (from ``chars``), the entire
    run is replaced by ``replacement``. Single occurrences are kept as-is.
    """
    if not text:
        return text

    result = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in chars:
            j = i + 1
            while j < n and text[j] == ch:
                j += 1
            if j - i >= 2:
                result.append(replacement)
            else:
                result.append(ch)
            i = j
        else:
            result.append(ch)
            i += 1
    return ''.join(result)


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace: collapse blank lines and multiple spaces."""
    # Collapse blank lines: strip each line, remove empty ones
    lines = text.split('\n')
    kept = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            kept.append(stripped)

    result = '\n'.join(kept)

    # Collapse all whitespace characters (including newlines) into single spaces
    chars = []
    prev_space = False
    for ch in result:
        if ch.isspace():
            if not prev_space:
                chars.append(' ')
                prev_space = True
        else:
            chars.append(ch)
            prev_space = False

    return ''.join(chars)
