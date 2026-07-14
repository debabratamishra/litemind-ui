from typing import Tuple, Optional
from .common import _is_word_char

def looks_like_url(text: str) -> bool:
    """Return True if the text appears to be a URL."""
    if not text:
        return False
    stripped = text.lstrip()
    lower_stripped = stripped.lower()
    return lower_stripped.startswith(("http://", "https://", "www."))


def clean_url(url: str) -> str:
    """Normalize a URL string that may contain stray spaces."""
    u = url.strip()

    if u.startswith("www."):
        u = "https://" + u

    u_lower = u.lower()
    for scheme in ("https", "http"):
        if u_lower.startswith(scheme):
            idx = len(scheme)
            while idx < len(u) and u[idx].isspace():
                idx += 1
            if idx < len(u) and u[idx] == ':':
                idx += 1
                while idx < len(u) and u[idx].isspace():
                    idx += 1
                if idx < len(u) and u[idx] == '/':
                    idx += 1
                    while idx < len(u) and u[idx].isspace():
                        idx += 1
                    if idx < len(u) and u[idx] == '/':
                        idx += 1
                        u = scheme + "://" + u[idx:]
                        break

    u = ".".join(segment.strip() for segment in u.split("."))
    u = "/".join(segment.strip() for segment in u.split("/"))
    u = ":".join(segment.strip() for segment in u.split(":"))

    parts = u.split()
    if len(parts) > 1:
        clean_url_result = parts[0]
        return "".join(clean_url_result.split())
    else:
        return "".join(u.split())


_URL_SOFT_CONTINUATION_CHARS = set(".:/?#[]@!$&'*+,;=%_-~")
_URL_HARD_BREAK_CHARS = set('<>"\'()[]{}|\\^`')
_TRAILING_URL_PUNCTUATION = ".,;:!?)"


def _looks_like_url_start(text: str, index: int) -> bool:
    """Return True when a URL-like token starts at ``index``."""
    if index > 0 and text[index - 1].isalnum():
        return False

    remaining = text[index:].lower()
    return remaining.startswith(("http://", "https://", "http:", "https:", "http :", "https :", "www."))


def _consume_url_candidate(text: str, start: int) -> Tuple[int, Optional[str]]:
    """Consume a URL-like substring, preserving trailing sentence punctuation."""
    index = start
    previous_non_space = ""

    while index < len(text):
        char = text[index]

        if char == "\n" or char in _URL_HARD_BREAK_CHARS:
            break

        if char.isspace():
            next_index = index
            while next_index < len(text) and text[next_index].isspace() and text[next_index] != "\n":
                next_index += 1

            if next_index >= len(text):
                break

            next_char = text[next_index]
            if next_char in _URL_SOFT_CONTINUATION_CHARS or previous_non_space in _URL_SOFT_CONTINUATION_CHARS:
                index = next_index
                continue

            break

        previous_non_space = char
        index += 1

    url_end = index
    while url_end > start and text[url_end - 1] in _TRAILING_URL_PUNCTUATION:
        url_end -= 1

    if url_end <= start:
        return start + 1, None

    cleaned = clean_url(text[start:url_end])
    if not looks_like_url(cleaned):
        return start + 1, None

    return url_end, cleaned


def _normalize_bare_urls(text: str) -> str:
    """Normalize bare and space-polluted URLs without broad regex matching."""
    normalized_parts = []
    cursor = 0
    index = 0

    while index < len(text):
        if _looks_like_url_start(text, index):
            url_end, cleaned = _consume_url_candidate(text, index)
            if cleaned:
                normalized_parts.append(text[cursor:index])
                normalized_parts.append(cleaned)
                cursor = url_end
                index = url_end
                continue

        index += 1

    normalized_parts.append(text[cursor:])
    return "".join(normalized_parts)


def _normalize_markdown_links(text: str) -> str:
    """Normalize Markdown links with a linear parser (no regex backtracking)."""
    normalized_parts = []
    cursor = 0
    index = 0
    text_length = len(text)

    while index < text_length:
        if text[index] != "[":
            index += 1
            continue

        link_start = index
        display_start = link_start + 1
        display_end = display_start

        while display_end < text_length and text[display_end] != "]":
            display_end += 1

        if display_end >= text_length:
            break

        if display_end == display_start or display_end + 1 >= text_length or text[display_end + 1] != "(":
            index = display_end + 1
            continue

        url_start = display_end + 2
        url_end = url_start
        while url_end < text_length and text[url_end] != ")":
            url_end += 1

        if url_end >= text_length:
            break

        if url_end == url_start:
            index = url_end + 1
            continue

        disp = text[display_start:display_end]
        url = text[url_start:url_end]
        clean_url_result = clean_url(url)
        clean_disp = clean_url(disp) if looks_like_url(disp) else disp

        normalized_parts.append(text[cursor:link_start])
        normalized_parts.append(f"[{clean_disp}]({clean_url_result})")

        cursor = url_end + 1
        index = cursor

    normalized_parts.append(text[cursor:])
    return "".join(normalized_parts)


def sanitize_links(text: str) -> str:
    """Fix common hyperlink formatting issues in Markdown text."""
    text = _normalize_markdown_links(text)
    return _normalize_bare_urls(text)
