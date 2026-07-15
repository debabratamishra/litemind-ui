"""Auto-enhancement: detect markdown patterns and convert them to ui:* blocks."""

import json
from typing import Optional

from .parsing import (
    _find_html_code_fence,
    _extract_primary_html_document,
    _find_primary_html_document_span,
)
from .webapp import _component_type_for_html


def _match_bold_kv_line(line: str) -> tuple[str, str] | None:
    """Extract (label, value) from a bold key-value line without regex.

    Handles::

        **Users:** 1,234
        **Users**: 1,234
        **Users** - 1,234
        - **Users:** 1,234
        * **Users:** 1,234
    """
    stripped = line.strip()
    if not stripped:
        return None

    # Skip optional markdown bullet prefix ("- " or "* ")
    text = stripped
    if len(text) >= 2 and text[0] in '-*' and text[1] == ' ':
        text = text[2:]

    # Must start with **
    if not text.startswith('**'):
        return None

    # Find the closing **
    close = text.find('**', 2)
    if close == -1:
        return None

    label = text[2:close]
    rest = text[close + 2:]  # everything after the closing **

    # Strip trailing separators and whitespace from the label
    label = label.rstrip(':- –')

    # Strip leading whitespace from the value side
    rest = rest.lstrip()

    # Strip optional separator (colon, dash, en-dash) and more whitespace
    if rest and rest[0] in ':-–':
        rest = rest[1:].lstrip()

    if not rest:
        return None

    return (label, rest)


def _split_md_row(line: str) -> list[str]:
    """Split a markdown table row into trimmed cell values."""
    stripped = line.strip().strip('|')
    return [cell.strip() for cell in stripped.split('|')]


def _looks_like_md_table_row(line: str) -> bool:
    """Return *True* when a line can participate in a markdown table."""
    stripped = line.strip()
    return bool(stripped) and '|' in stripped


def _is_md_table_separator_cell(cell: str) -> bool:
    """Return *True* for ``---`` / ``:---:`` style markdown separator cells."""
    stripped = cell.strip()
    if not stripped:
        return False
    if stripped.startswith(':'):
        stripped = stripped[1:]
    if stripped.endswith(':'):
        stripped = stripped[:-1]
    return len(stripped) >= 2 and all(char == '-' for char in stripped)


def _is_md_table_separator_row(line: str) -> bool:
    """Return *True* when a line is a markdown table separator row."""
    cells = [cell for cell in _split_md_row(line) if cell]
    return bool(cells) and all(_is_md_table_separator_cell(cell) for cell in cells)


def _parse_md_table(table_text: str) -> Optional[dict]:
    """Parse a markdown table string into ``data_table`` props."""
    lines = [l.strip() for l in table_text.strip().splitlines() if l.strip()]
    if len(lines) < 3:
        return None

    # Validate the separator row (line 1)
    if not _is_md_table_separator_row(lines[1]):
        return None

    columns = _split_md_row(lines[0])
    data = [_split_md_row(line) for line in lines[2:]]

    if not columns or not data:
        return None

    # Normalise column counts – pad shorter rows with empty strings
    max_cols = max(len(columns), max((len(row) for row in data), default=0))
    columns = columns + [''] * (max_cols - len(columns))
    data = [row + [''] * (max_cols - len(row)) for row in data]

    return {"columns": columns, "data": data}


def _auto_convert_tables(text: str) -> str:
    """Replace markdown tables with ``ui:data_table`` blocks."""
    lines = text.splitlines(keepends=True)
    if len(lines) < 3:
        return text

    converted: list[str] = []
    index = 0
    total_lines = len(lines)

    while index < total_lines:
        if index + 2 >= total_lines:
            converted.append(lines[index])
            index += 1
            continue

        if not _looks_like_md_table_row(lines[index]):
            converted.append(lines[index])
            index += 1
            continue

        if not _is_md_table_separator_row(lines[index + 1]):
            converted.append(lines[index])
            index += 1
            continue

        if not _looks_like_md_table_row(lines[index + 2]):
            converted.append(lines[index])
            index += 1
            continue

        table_end = index + 3
        while table_end < total_lines and _looks_like_md_table_row(lines[table_end]):
            table_end += 1

        table_text = ''.join(lines[index:table_end])
        props = _parse_md_table(table_text)
        if props is None:
            converted.append(lines[index])
            index += 1
            continue

        converted.append(f'\n```ui:data_table\n{json.dumps(props)}\n```\n')
        index = table_end

    return ''.join(converted)


def _auto_convert_metrics(text: str) -> str:
    """Convert consecutive **Label:** Value lines into ``ui:metric`` blocks."""
    lines = text.split('\n')
    result_lines: list = []
    group: list = []
    group_start = -1

    def _flush():
        nonlocal group, group_start
        if len(group) >= 2:
            props = {"metrics": group[:4]}
            result_lines.append(f'```ui:metric\n{json.dumps(props)}\n```')
        else:
            result_lines.extend(lines[group_start:group_start + len(group)])
        group = []
        group_start = -1

    for i, line in enumerate(lines):
        m = _match_bold_kv_line(line)
        if m:
            if not group:
                group_start = i
            group.append({"label": m[0].rstrip(":- –").strip(), "value": m[1].strip()})
        else:
            if group:
                _flush()
            result_lines.append(line)
    if group:
        _flush()

    return '\n'.join(result_lines)


def _wrap_html_markup_as_ui_block(markup: str) -> str:
    """Wrap standalone HTML markup in a ui:* fenced block."""
    stripped = markup.strip()
    if not stripped:
        return ""
    component_type = _component_type_for_html(stripped)
    return f"```ui:{component_type}\n{stripped}\n```"


def _auto_convert_html_markup(text: str) -> str:
    """Convert bare HTML documents or ```html fences into ui:* blocks."""
    stripped = text.strip()
    if not stripped:
        return text

    html_fence = _find_html_code_fence(stripped)
    if html_fence is not None:
        fence_start, fence_end, fence_body = html_fence
        if fence_start == 0 and fence_end == len(stripped):
            return _wrap_html_markup_as_ui_block(fence_body)

    document = _extract_primary_html_document(stripped)
    if document is not None and document == stripped:
        return _wrap_html_markup_as_ui_block(document)

    html_fence = _find_html_code_fence(text)
    if html_fence is not None:
        fence_start, fence_end, fence_body = html_fence
        before = text[:fence_start]
        after = text[fence_end:]
        return before + _wrap_html_markup_as_ui_block(fence_body) + after

    document_span = _find_primary_html_document_span(text)
    if document_span is not None:
        start, end = document_span
        before = text[:start]
        after = text[end:]
        return before + _wrap_html_markup_as_ui_block(text[start:end]) + after

    return text


def auto_enhance_content(text: str) -> str:
    """Convert common markdown patterns to ``ui:*`` blocks.

    Called when the Generative UI toggle is enabled but the model did not
    emit any native ``ui:*`` fenced blocks (typical for smaller models).
    Converts markdown tables → ``ui:data_table`` and bold key-value
    metric lines → ``ui:metric``.
    """
    text = _auto_convert_html_markup(text)
    text = _auto_convert_tables(text)
    text = _auto_convert_metrics(text)
    return text
