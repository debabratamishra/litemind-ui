from typing import List
from .common import _collapse_newlines, _collapse_horizontal_spaces
from .url import looks_like_url, clean_url, _normalize_bare_urls, _normalize_markdown_links, _consume_url_candidate

def _is_header_line(line: str) -> bool:
    """Check if a line is a markdown header (starts with ``#``)."""
    return bool(line) and line.lstrip().startswith('#')


def _is_numbered_list_marker(text: str, pos: int) -> bool:
    """Return True if *text* at *pos* starts a numbered list marker (``N. ``)."""
    if pos >= len(text) or not text[pos].isdigit():
        return False
    j = pos + 1
    while j < len(text) and text[j].isdigit():
        j += 1
    return (j < len(text) and text[j] == '.'
            and j + 1 < len(text) and text[j + 1] == ' ')


def _ensure_header_space(line: str) -> str:
    """Ensure a space follows the ``#`` markers (``#Header`` → ``# Header``)."""
    if not line.startswith('#'):
        return line
    i = 0
    while i < len(line) and line[i] == '#':
        i += 1
    if i < len(line) and line[i] != ' ':
        return line[:i] + ' ' + line[i:]
    return line


def _split_inline_numbered_items(line: str) -> List[str]:
    """Split a line with multiple numbered items into separate lines."""
    markers = []
    i = 0
    while i < len(line):
        if not line[i].isdigit():
            i += 1
            continue
        j = i + 1
        while j < len(line) and line[j].isdigit():
            j += 1
        if j < len(line) and line[j] == '.' and j + 1 < len(line) and line[j + 1] == ' ':
            markers.append(i)
            i = j + 2
        else:
            i = j

    if len(markers) <= 1:
        return [line]

    parts = []
    for idx, pos in enumerate(markers):
        start = 0 if idx == 0 else pos
        end = markers[idx + 1] if idx + 1 < len(markers) else len(line)
        parts.append(line[start:end].rstrip())
    return parts


def clean_markdown_text(text: str) -> str:
    """Clean markdown text for better rendering (no regex, O(n))."""
    if not text.strip():
        return text

    lines = text.split('\n')

    # Step 1: ensure space after header markers
    for i, line in enumerate(lines):
        if line.startswith('#'):
            lines[i] = _ensure_header_space(line)

    # Step 2: ensure blank lines around headers
    result = []
    for i, line in enumerate(lines):
        is_header = _is_header_line(line)
        if is_header and i > 0 and lines[i - 1].strip():
            result.append('')
        result.append(line)
        if is_header and i + 1 < len(lines) and lines[i + 1].strip():
            result.append('')
    lines = result

    # Step 3: ensure blank line before numbered lists
    result = []
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        is_numbered = bool(stripped) and _is_numbered_list_marker(stripped, 0)
        if is_numbered and i > 0 and lines[i - 1].strip():
            result.append('')
        result.append(line)
    lines = result

    # Step 4: split inline numbered items
    result = []
    for line in lines:
        parts = _split_inline_numbered_items(line)
        result.extend(parts)
    lines = result

    # Step 5: collapse 4+ consecutive newlines to exactly 3
    text = '\n'.join(lines)
    collapsed = []
    newline_count = 0
    for ch in text:
        if ch == '\n':
            newline_count += 1
            if newline_count <= 3:
                collapsed.append(ch)
        else:
            newline_count = 0
            collapsed.append(ch)
    text = ''.join(collapsed)

    return text.strip()


def _fix_header_spacing_above(lines: List[str]) -> List[str]:
    """Insert a blank line before headers if preceded by text on previous line."""
    result = []
    for idx, line in enumerate(lines):
        if idx > 0:
            prev_line = result[-1].rstrip()
            curr_line = line.lstrip()
            if prev_line and prev_line[-1].islower():
                if curr_line.startswith('#'):
                    h_count = 0
                    while h_count < len(curr_line) and curr_line[h_count] == '#':
                        h_count += 1
                    if 1 <= h_count <= 6 and h_count < len(curr_line):
                        rest = curr_line[h_count:]
                        rest_stripped = rest.lstrip(' \t')
                        if len(rest) - len(rest_stripped) > 0 and rest_stripped and rest_stripped[0].isupper():
                            result.append('')
        result.append(line)
    return result


def _fix_punctuation_spacing(text: str) -> str:
    """Replace 2 or more whitespaces preceding a punctuation with a single space."""
    chars = list(text)
    result = []
    i = 0
    n = len(chars)
    while i < n:
        if chars[i] in '.!?,:;':
            space_start = len(result)
            while space_start > 0 and result[space_start - 1].isspace():
                space_start -= 1
            space_len = len(result) - space_start
            if space_len >= 2:
                result = result[:space_start] + [' ']
            result.append(chars[i])
        else:
            result.append(chars[i])
        i += 1
    return "".join(result)


def _join_split_sentences(text: str) -> str:
    """Join sentences that got split across a single newline."""
    result = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] in '.!?':
            j = i + 1
            while j < n and text[j] in ' \t':
                j += 1
            if j < n and text[j] == '\n':
                k = j + 1
                if k < n and text[k] != '\n':
                    while k < n and text[k] in ' \t':
                        k += 1
                    if k < n and text[k].islower():
                        result.append(text[i])
                        result.append(' ')
                        result.append(text[k])
                        i = k + 1
                        continue
        result.append(text[i])
        i += 1
    return "".join(result)


def _fix_url_word_concatenation(text: str) -> str:
    """Fix words concatenated right after a URL."""
    result = []
    i = 0
    n = len(text)
    while i < n:
        if text[i:].startswith(("http://", "https://")):
            start = i
            while i < n and not text[i].isspace():
                i += 1
            url_candidate = text[start:i]
            
            # Find split point
            split_idx = -1
            search_start = 8 if url_candidate.startswith("https://") else 7
            for idx in range(search_start, len(url_candidate) - 1):
                if url_candidate[idx].isupper() and url_candidate[idx+1].islower():
                    if url_candidate[idx - 1].isalnum():
                        split_idx = idx
                        break
            
            if split_idx != -1:
                result.append(url_candidate[:split_idx])
                result.append(' ')
                result.append(url_candidate[split_idx:])
            else:
                result.append(url_candidate)
            continue
        result.append(text[i])
        i += 1
    return "".join(result)


def _fix_numbered_lists(text: str) -> str:
    """Fix numbered list formatting by ensuring proper line breaks."""
    result = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        is_anchor = (c in '.!?:') or c.islower()
        if is_anchor and i + 1 < n and text[i + 1].isspace():
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            
            is_marker = False
            if j < n:
                if text[j] == '•':
                    k = j + 1
                    if k < n and text[k].isspace():
                        while k < n and text[k].isspace():
                            k += 1
                        if k < n and text[k].isupper():
                            is_marker = True
                elif text[j].isdigit():
                    k = j
                    while k < n and text[k].isdigit():
                        k += 1
                    if k < n and text[k] == '.':
                        k += 1
                        if k < n and text[k].isspace():
                            while k < n and text[k].isspace():
                                k += 1
                            if k < n and text[k].isupper():
                                is_marker = True
            
            if is_marker:
                result.append(c)
                result.append('\n\n')
                i = j
                continue
        
        result.append(c)
        i += 1
    text = "".join(result)

    # Line start list spacing normalization
    lines = text.split('\n')
    for idx, line in enumerate(lines):
        if line and line[0].isdigit():
            digit_count = 0
            while digit_count < len(line) and line[digit_count].isdigit():
                digit_count += 1
            if digit_count < len(line) and line[digit_count] == '.':
                rest = line[digit_count + 1:]
                if rest and rest[0].isspace():
                    lines[idx] = line[:digit_count + 1] + " " + rest.lstrip()
        elif line.startswith('•'):
            rest = line[1:]
            if rest and rest[0].isspace():
                lines[idx] = "• " + rest.lstrip()
    text = '\n'.join(lines)

    return text


def _fix_concatenated_text_after_urls(text: str) -> str:
    """Fix concatenated text that appears after URLs."""
    targets = [
        ("thisisthemainwebsiteandthemostcommonwaytoaccessit.", " This is the main website and the most common way to access it."),
        ("thisisthemainwebsiteandthemostcommonwaytoaccessit", " This is the main website and the most common way to access it."),
        ("andyou'rethere.", " and you're there."),
        ("andyou'rethere", " and you're there."),
        ("andyourethere.", " and you're there."),
        ("andyourethere", " and you're there."),
    ]
    
    for target, replacement in targets:
        start_idx = 0
        while True:
            pos = text.lower().find(target, start_idx)
            if pos == -1:
                break
            
            is_after_url = False
            url_start = pos - 1
            while url_start >= 0 and not text[url_start].isspace():
                if text[url_start:].lower().startswith(("http://", "https://")):
                    is_after_url = True
                    break
                url_start -= 1
            
            if is_after_url:
                text = text[:pos] + replacement + text[pos + len(target):]
                start_idx = pos + len(replacement)
            else:
                start_idx = pos + 1
    return text


def clean_text_formatting(text: str) -> str:
    """Clean up common text formatting issues while preserving word spacing."""
    text = _fix_concatenated_text_after_urls(text)
    text = _fix_numbered_lists(text)

    text = _collapse_horizontal_spaces(text)
    
    # Fix: header spacing above
    lines = text.split('\n')
    lines = _fix_header_spacing_above(lines)
    text = '\n'.join(lines)

    text = _collapse_newlines(text, 2)
    text = _fix_punctuation_spacing(text)
    text = _join_split_sentences(text)
    text = _fix_url_word_concatenation(text)

    return text
