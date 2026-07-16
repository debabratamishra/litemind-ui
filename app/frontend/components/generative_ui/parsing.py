"""Fenced-block and embedded-HTML parsing helpers (pure, no Streamlit)."""

import json
from typing import Iterator, Optional


def _is_valid_fence_info(info: str) -> bool:
    """Return *True* when *info* matches the supported fenced-code syntax."""
    return all(char.isalnum() or char == "_" or char in ":.-" for char in info)


def _is_valid_ui_component_type(component_type: str) -> bool:
    """Return *True* for supported ``ui:<component>`` fence tags."""
    return bool(component_type) and all(
        char.isalnum() or char == "_" for char in component_type
    )


def _iter_fenced_blocks(text: str) -> Iterator[tuple[int, int, str, str]]:
    """Yield complete fenced blocks as ``(start, end, lang, content)`` tuples."""
    search_start = 0
    text_length = len(text)

    while search_start < text_length:
        fence_start = text.find("```", search_start)
        if fence_start == -1:
            return

        header_end = text.find("\n", fence_start + 3)
        if header_end == -1:
            return

        lang = text[fence_start + 3 : header_end].strip()
        if lang and not _is_valid_fence_info(lang):
            search_start = fence_start + 3
            continue

        fence_end = text.find("```", header_end + 1)
        if fence_end == -1:
            return

        yield fence_start, fence_end + 3, lang, text[header_end + 1 : fence_end]
        search_start = fence_end + 3


def _extract_fenced_body(text: str) -> str:
    """Return fenced body when *text* is exactly one fenced block."""
    stripped = text.strip()
    block = _match_single_fenced_block(stripped)
    if block is None:
        return stripped
    _, body = block
    return body.strip()


def _match_single_fenced_block(text: str) -> Optional[tuple[str, str]]:
    block_iterator = _iter_fenced_blocks(text)
    block = next(block_iterator, None)
    if block is None:
        return None

    start, end, lang, content = block
    if start != 0 or end != len(text):
        return None

    return lang, content


def _find_html_code_fence(text: str) -> Optional[tuple[int, int, str]]:
    for fence_start, fence_end, lang, content in _iter_fenced_blocks(text):
        if lang.casefold() == "html":
            return fence_start, fence_end, content
    return None


def _try_parse_json_object(text: str) -> Optional[dict]:
    """Best-effort JSON object parse; returns None when parsing fails."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _is_html_tag_boundary(char: str) -> bool:
    return char.isspace() or char == ">"


def _parse_doctype_html_tag_end(text: str, lowered: str, start: int) -> tuple[Optional[int], int]:
    token_end = start + len("<!doctype")
    if token_end >= len(text) or not text[token_end].isspace():
        return None, start + 1

    index = token_end
    while index < len(text):
        char = text[index]
        if char == ">":
            body = lowered[token_end:index].lstrip()
            if not body.startswith("html"):
                return None, index + 1
            if len(body) > 4 and (body[4].isalnum() or body[4] == "_"):
                return None, index + 1
            return index + 1, index + 1
        if char == "<":
            return None, index
        index += 1

    return None, len(text)


def _parse_html_open_tag_end(text: str, start: int) -> tuple[Optional[int], int]:
    token_end = start + len("<html")
    if token_end < len(text) and not _is_html_tag_boundary(text[token_end]):
        return None, start + 1

    index = token_end
    while index < len(text):
        char = text[index]
        if char == ">":
            return index + 1, index + 1
        if char == "<":
            return None, index
        index += 1

    return None, len(text)


def _parse_html_close_tag_end(text: str, start: int) -> tuple[Optional[int], int]:
    token_end = start + len("</html")
    if token_end < len(text) and not _is_html_tag_boundary(text[token_end]):
        return None, start + 1

    index = token_end
    while index < len(text):
        char = text[index]
        if char == ">":
            return index + 1, index + 1
        if char == "<":
            return None, index
        if not char.isspace():
            return None, index + 1
        index += 1

    return None, len(text)


def _find_primary_html_document_span(text: str) -> Optional[tuple[int, int]]:
    lowered = text.lower()
    search_start = 0
    pending_doctype_start: Optional[int] = None

    while True:
        tag_start = text.find("<", search_start)
        if tag_start == -1:
            return None

        if lowered.startswith("<!doctype", tag_start):
            doctype_end, next_index = _parse_doctype_html_tag_end(text, lowered, tag_start)
            if doctype_end is not None and pending_doctype_start is None:
                pending_doctype_start = tag_start
                search_start = doctype_end
                continue

            search_start = max(next_index, tag_start + 1)
            continue

        if lowered.startswith("<html", tag_start):
            open_tag_end, next_index = _parse_html_open_tag_end(text, tag_start)
            if open_tag_end is None:
                search_start = max(next_index, tag_start + 1)
                continue

            close_search_start = open_tag_end
            while True:
                close_tag_start = text.find("<", close_search_start)
                if close_tag_start == -1:
                    return None

                if lowered.startswith("</html", close_tag_start):
                    close_tag_end, next_close_index = _parse_html_close_tag_end(text, close_tag_start)
                    if close_tag_end is not None:
                        start = pending_doctype_start if pending_doctype_start is not None else tag_start
                        return start, close_tag_end

                    close_search_start = max(next_close_index, close_tag_start + 1)
                    continue

                close_search_start = close_tag_start + 1

        search_start = tag_start + 1


def _extract_primary_html_document(text: str) -> Optional[str]:
    """Return the first complete HTML document embedded in *text*, if present."""
    document_span = _find_primary_html_document_span(text)
    if document_span is None:
        return None
    start, end = document_span
    return text[start:end].strip()
