"""Helpers for stripping lightweight markup without regex backtracking risks."""

from typing import Iterable


def _match_tag(text: str, start: int, allowed_tags: set[str]) -> tuple[str, bool, int] | None:
    """Parse a simple HTML-like tag at ``start`` if it is in ``allowed_tags``."""
    if start >= len(text) or text[start] != "<":
        return None

    index = start + 1
    while index < len(text) and text[index].isspace():
        index += 1

    is_closing = False
    if index < len(text) and text[index] == "/":
        is_closing = True
        index += 1
        while index < len(text) and text[index].isspace():
            index += 1

    name_start = index
    while index < len(text) and text[index].isalpha():
        index += 1

    if index == name_start:
        return None

    tag_name = text[name_start:index].lower()
    if tag_name not in allowed_tags:
        return None

    while index < len(text) and text[index].isspace():
        index += 1

    if index >= len(text) or text[index] != ">":
        return None

    return tag_name, is_closing, index + 1


def _find_tagged_spans(text: str, tag_names: Iterable[str]) -> list[tuple[int, int, str]]:
    """Return top-level matched tag spans and their inner content."""
    allowed_tags = {tag.lower() for tag in tag_names}
    stack: list[tuple[str, int, int]] = []
    spans: list[tuple[int, int, str]] = []
    index = 0

    while index < len(text):
        match = _match_tag(text, index, allowed_tags)
        if not match:
            index += 1
            continue

        tag_name, is_closing, tag_end = match
        if is_closing:
            if stack and stack[-1][0] == tag_name:
                _, open_start, content_start = stack.pop()
                spans.append((open_start, tag_end, text[content_start:index]))
        else:
            stack.append((tag_name, index, tag_end))

        index = tag_end

    spans.sort(key=lambda span: (span[0], -(span[1] - span[0])))

    top_level_spans: list[tuple[int, int, str]] = []
    for span in spans:
        if top_level_spans and span[0] < top_level_spans[-1][1]:
            continue
        top_level_spans.append(span)

    return top_level_spans


def extract_tagged_sections(text: str, tag_names: Iterable[str]) -> tuple[list[str], str]:
    """Extract top-level tagged content and return it with the remaining text."""
    spans = _find_tagged_spans(text, tag_names)
    if not spans:
        return [], text

    extracted: list[str] = []
    cleaned_parts: list[str] = []
    cursor = 0

    for start, end, content in spans:
        cleaned_parts.append(text[cursor:start])
        stripped_content = content.strip()
        if stripped_content:
            extracted.append(stripped_content)
        cursor = end

    cleaned_parts.append(text[cursor:])
    return extracted, "".join(cleaned_parts)


def remove_tagged_sections(text: str, tag_names: Iterable[str]) -> str:
    """Remove simple tagged sections from text."""
    _, cleaned_text = extract_tagged_sections(text, tag_names)
    return cleaned_text


def replace_fenced_code_blocks(text: str, replacement: str) -> str:
    """Replace complete triple-backtick code fences without regex scanning."""
    if "```" not in text:
        return text

    parts: list[str] = []
    cursor = 0

    while True:
        fence_start = text.find("```", cursor)
        if fence_start == -1:
            parts.append(text[cursor:])
            break

        fence_end = text.find("```", fence_start + 3)
        if fence_end == -1:
            parts.append(text[cursor:])
            break

        parts.append(text[cursor:fence_start])
        parts.append(replacement)
        cursor = fence_end + 3

    return "".join(parts)
