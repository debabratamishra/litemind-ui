"""
Generative UI components for dynamic rendering in chat responses.

Inspired by the AG-UI (Agent-User Interaction) protocol's event-driven
component model. Renders rich, interactive UI elements from structured
specifications embedded in LLM output.

Protocol: LLM emits fenced code blocks with a ``ui:<component>`` language
tag. Most components use a JSON props object. ``ui:webapp`` is the one
exception: its block body is raw HTML/CSS/JS that is rendered inside a
Streamlit iframe for interactive mini-apps.
"""

import base64
import html
import json
import logging
import re
import streamlit as st
from typing import Any, Callable, Dict, Iterator, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fence parsing helpers
# ---------------------------------------------------------------------------

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

_WEBAPP_HEIGHT_RE = re.compile(r"^\s*<!--\s*height\s*:\s*(\d{2,4})\s*-->\s*", re.IGNORECASE)

_WEBAPP_CSS = """
<style>
/* Make sure the chat message content itself doesn't block events */
[data-testid="stChatMessageContent"] {
    pointer-events: auto !important;
}

/* Ensure iframe containers and the iframes themselves are interactive */
[data-testid="stIFrame"],
[data-testid="stIFrame"] > iframe,
[data-testid="stCustomComponentV1"],
[data-testid="stCustomComponentV1"] > iframe {
    pointer-events: auto !important;
}

/* Disable interaction on Streamlit toolbars that might overlap the iframe */
div[data-testid="stElementToolbar"],
div[data-testid="stElementToolbar"] * {
    pointer-events: none !important;
    display: none !important;
}
</style>
"""


def has_ui_blocks(text: str) -> bool:
    """Return *True* if *text* contains at least one complete UI block."""
    for _, _, lang, _ in _iter_fenced_blocks(text):
        if lang.startswith("ui:") and _is_valid_ui_component_type(lang[3:]):
            return True
    return False


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

def _sanitize_html(text: str) -> str:
    """HTML-escape user/LLM-provided text before embedding in markup."""
    return html.escape(str(text)) if text else ""


def _sanitize_url(url: str) -> str:
    """Reject dangerous URL schemes and escape for safe embedding."""
    url = str(url).strip()
    if url.lower().startswith("javascript:"):
        return "#"
    if not url.startswith(("http://", "https://", "/")):
        return "#"
    return html.escape(url)


# ---------------------------------------------------------------------------
# Component registry & dispatcher
# ---------------------------------------------------------------------------

def render_ui_component(
    component_type: str,
    props: Any,
    msg_index: int = 0,
) -> bool:
    """Render a single UI component.  Returns *True* on success."""
    renderer = _COMPONENT_REGISTRY.get(component_type)
    if renderer is None:
        st.warning(f"Unknown UI component: `{component_type}`")
        return False
    try:
        renderer(props, msg_index)
        return True
    except Exception as exc:
        logger.error("Error rendering UI component '%s': %s", component_type, exc)
        st.error(f"Error rendering component: {exc}")
        return False


def render_mixed_content(text: str, msg_index: int = 0) -> None:
    """Render text that may interleave markdown, code blocks, and UI blocks."""
    from ..utils.text_processing import (
        clean_markdown_text,
        sanitize_links,
        unescape_text,
    )

    pos = 0
    rendered_any = False

    for fence_start, fence_end, lang, content in _iter_fenced_blocks(text):
        # --- text segment before this fence ---
        before = text[pos:fence_start]
        if before.strip():
            # Don't apply clean_text_formatting here – it was already
            # skipped at the top level for generative-UI content and
            # re-applying it can break markdown tables in mixed content.
            cleaned = clean_markdown_text(before)
            st.markdown(sanitize_links(unescape_text(cleaned)))
            rendered_any = True

        ui_content = content.strip()

        if lang.startswith("ui:"):
            component_type = lang[3:]
            try:
                if component_type == "webapp":
                    render_ui_component(component_type, ui_content, msg_index)
                    rendered_any = True
                    pos = fence_end
                    continue

                # Streaming can inject line-breaks inside the JSON body
                # (e.g. Ollama flushes on sentence-ending punctuation like
                # periods inside string values).  Collapse whitespace so
                # that json.loads() can still parse the payload.
                normalised = ' '.join(ui_content.split())
                props = json.loads(normalised)
                render_ui_component(component_type, props, msg_index)
            except json.JSONDecodeError:
                st.warning(f"Invalid JSON in UI component `{component_type}`")
                st.code(ui_content, language="json")
            rendered_any = True
        else:
            # Regular code block
            code = content.strip("\n")
            st.code(code, language=lang if lang else None)
            rendered_any = True

        pos = fence_end

    # --- trailing text ---
    tail = text[pos:]
    if tail.strip() or not rendered_any:
        cleaned = clean_markdown_text(tail)
        st.markdown(sanitize_links(unescape_text(cleaned)))


# ===================================================================
# Component renderers
# ===================================================================

def _render_info_card(props: dict, msg_index: int = 0) -> None:
    icon = _sanitize_html(props.get("icon", "ℹ️"))
    title = _sanitize_html(props.get("title", ""))
    content = _sanitize_html(props.get("content", ""))
    color = _sanitize_html(props.get("color", "#1f77b4"))

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {color}15, {color}08);
            padding: 1rem 1.2rem;
            border-radius: 0.75rem;
            border-left: 4px solid {color};
            margin: 0.5rem 0;
        ">
            <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.4rem;">
                {icon} {title}
            </div>
            <div style="font-size: 0.95rem; line-height: 1.5;">
                {content}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_data_table(props: dict, msg_index: int = 0) -> None:
    import pandas as pd

    title = props.get("title", "")
    columns = props.get("columns", [])
    data = props.get("data", [])

    if title:
        st.markdown(f"**{_sanitize_html(title)}**")

    if columns and data:
        df = pd.DataFrame(data, columns=columns)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No data to display.")


def _render_metric(props: dict, msg_index: int = 0) -> None:
    metrics = props.get("metrics", [props])
    if not isinstance(metrics, list):
        metrics = [metrics]

    cols = st.columns(min(len(metrics), 4))
    for col, metric in zip(cols, metrics):
        with col:
            st.metric(
                label=str(metric.get("label", "")),
                value=str(metric.get("value", "")),
                delta=metric.get("delta"),
                delta_color=metric.get("delta_color", "normal"),
            )


def _render_chart(props: dict, msg_index: int = 0) -> None:
    chart_type = props.get("type", "bar")
    title = props.get("title", "")
    x = props.get("x", [])
    y = props.get("y", [])

    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        series = props.get("series")

        if series:
            for s in series:
                tx, ty = s.get("x", x), s.get("y", [])
                name = s.get("name", "")
                if chart_type == "bar":
                    fig.add_trace(go.Bar(x=tx, y=ty, name=name))
                elif chart_type in ("line", "scatter"):
                    mode = "lines+markers" if chart_type == "line" else "markers"
                    fig.add_trace(go.Scatter(x=tx, y=ty, mode=mode, name=name))
        else:
            if chart_type == "bar":
                fig.add_trace(go.Bar(x=x, y=y))
            elif chart_type == "line":
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers"))
            elif chart_type == "pie":
                fig.add_trace(go.Pie(labels=x, values=y))
            elif chart_type == "scatter":
                fig.add_trace(go.Scatter(x=x, y=y, mode="markers"))

        fig.update_layout(
            title=title,
            xaxis_title=props.get("x_label", ""),
            yaxis_title=props.get("y_label", ""),
            template="plotly_dark",
            height=props.get("height", 400),
            margin=dict(l=40, r=40, t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        # Fallback to built-in Streamlit charts
        import pandas as pd

        if title:
            st.markdown(f"**{_sanitize_html(title)}**")
        if x and y:
            df = pd.DataFrame({"x": x, "y": y}).set_index("x")
            if chart_type == "bar":
                st.bar_chart(df)
            else:
                st.line_chart(df)
        else:
            st.info("Chart data unavailable.")


def _render_button_group(props: dict, msg_index: int = 0) -> None:
    label = props.get("label", "")
    buttons = props.get("buttons", [])

    if label:
        st.markdown(f"**{_sanitize_html(label)}**")

    if not buttons:
        return

    cols = st.columns(min(len(buttons), 4))
    for i, (col, btn) in enumerate(zip(cols, buttons)):
        with col:
            btn_text = str(btn.get("text", f"Option {i + 1}"))
            btn_value = str(btn.get("value", btn_text))
            btn_key = f"genui_btn_{msg_index}_{i}_{hash(btn_text) % 100000}"

            if st.button(btn_text, key=btn_key, use_container_width=True):
                st.session_state["genui_pending_input"] = btn_value


def _render_progress(props: dict, msg_index: int = 0) -> None:
    value = props.get("value", 0)
    label = props.get("label", "")

    if label:
        st.markdown(f"**{_sanitize_html(label)}**")
    st.progress(min(max(int(value) / 100.0, 0.0), 1.0))


def _render_alert(props: dict, msg_index: int = 0) -> None:
    level = props.get("level", "info")
    message = str(props.get("message", ""))

    dispatch = {
        "success": st.success,
        "warning": st.warning,
        "error": st.error,
        "info": st.info,
    }
    dispatch.get(level, st.info)(message)


def _render_columns(props: dict, msg_index: int = 0) -> None:
    items = props.get("items", [])
    if not items:
        return

    cols = st.columns(min(len(items), 4))
    for col, item in zip(cols, items):
        with col:
            icon = item.get("icon", "")
            title_text = item.get("title", "")
            content = item.get("content", "")
            if icon:
                st.markdown(f"### {_sanitize_html(icon)}")
            if title_text:
                st.markdown(f"**{_sanitize_html(title_text)}**")
            if content:
                st.markdown(content)


def _render_json_viewer(props: dict, msg_index: int = 0) -> None:
    title = props.get("title", "")
    data = props.get("data", {})
    expanded = props.get("expanded", True)

    if title:
        with st.expander(_sanitize_html(title), expanded=expanded):
            st.json(data)
    else:
        st.json(data)


def _render_link_cards(props: dict, msg_index: int = 0) -> None:
    links = props.get("links", [])
    if not links:
        return

    cols = st.columns(min(len(links), 3))
    for i, link in enumerate(links):
        with cols[i % 3]:
            title = _sanitize_html(link.get("title", "Link"))
            url = _sanitize_url(link.get("url", "#"))
            description = _sanitize_html(link.get("description", ""))

            st.markdown(
                f"""
                <a href="{url}" target="_blank" style="text-decoration: none;">
                    <div style="
                        background: rgba(255,255,255,0.05);
                        border: 1px solid rgba(255,255,255,0.1);
                        border-radius: 0.5rem;
                        padding: 0.8rem;
                        margin: 0.3rem 0;
                    ">
                        <div style="font-weight: 600; color: #58a6ff;">{title}</div>
                        <div style="font-size: 0.85rem; color: #8b949e; margin-top: 0.3rem;">
                            {description}
                        </div>
                    </div>
                </a>
                """,
                unsafe_allow_html=True,
            )


def _render_steps(props: dict, msg_index: int = 0) -> None:
    steps = props.get("steps", [])
    current = props.get("current", 0)

    for i, step in enumerate(steps):
        if isinstance(step, str):
            label, desc = step, ""
        else:
            label = step.get("label", f"Step {i + 1}")
            desc = step.get("description", "")

        icon = "✅" if i < current else ("🔵" if i == current else "⬜")
        line = f"{icon} **{_sanitize_html(label)}**"
        if desc:
            line += f"  \n{_sanitize_html(desc)}"
        st.markdown(line)


def _render_tabs(props: dict, msg_index: int = 0) -> None:
    tabs_data = props.get("tabs", [])
    if not tabs_data:
        return

    tab_labels = [t.get("label", f"Tab {i + 1}") for i, t in enumerate(tabs_data)]
    tabs = st.tabs(tab_labels)

    for tab, tab_data in zip(tabs, tabs_data):
        with tab:
            st.markdown(tab_data.get("content", ""))


def _render_callout(props: dict, msg_index: int = 0) -> None:
    emoji = _sanitize_html(props.get("emoji", "💡"))
    title = _sanitize_html(props.get("title", ""))
    content = _sanitize_html(props.get("content", ""))
    color = _sanitize_html(props.get("color", "#ffd700"))

    title_html = f"<strong>{emoji} {title}</strong><br/>" if title else f"{emoji} "

    st.markdown(
        f"""
        <div style="
            background: {color}12;
            border: 1px solid {color}40;
            border-radius: 0.5rem;
            padding: 0.8rem 1rem;
            margin: 0.5rem 0;
        ">
            {title_html}{content}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _inject_webapp_css() -> None:
    if st.session_state.get("_genui_webapp_css_injected"):
        return

    st.markdown(_WEBAPP_CSS, unsafe_allow_html=True)
    st.session_state["_genui_webapp_css_injected"] = True


def _clamp_webapp_height(height: int) -> int:
    return max(240, min(height, 2000))


def _extract_webapp_height(html_content: str, default_height: int = 520) -> tuple[int, str]:
    match = _WEBAPP_HEIGHT_RE.match(html_content)
    if match is None:
        return default_height, html_content.strip()

    hinted_height = _clamp_webapp_height(int(match.group(1)))
    return hinted_height, html_content[match.end():].strip()


def _wrap_webapp_html(html_content: str) -> str:
    stripped = html_content.strip()
    if not stripped:
        return ""

    if re.search(r"<html[\s>]", stripped, re.IGNORECASE):
        return stripped

    if re.search(r"<(head|body)[\s>]", stripped, re.IGNORECASE):
        return (
            "<!DOCTYPE html>\n"
            "<html>\n"
            f"{stripped}\n"
            "</html>"
        )

    return (
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head>\n"
        "  <meta charset=\"utf-8\" />\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />\n"
        "  <style>\n"
        "    html, body { margin: 0; padding: 0; }\n"
        "    body { font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif; padding: 16px; }\n"
        "    * { box-sizing: border-box; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        f"{stripped}\n"
        "</body>\n"
        "</html>"
    )


def _build_webapp_iframe_src(html_content: str) -> str:
    wrapped_html = _wrap_webapp_html(html_content)
    if not wrapped_html:
        return ""

    encoded_html = base64.b64encode(wrapped_html.encode("utf-8")).decode("ascii")
    return f"data:text/html;base64,{encoded_html}"


def _render_webapp(props: Any, msg_index: int = 0) -> None:
    explicit_height = None
    if isinstance(props, dict):
        html_content = str(props.get("html", "")).strip()
        height_value = props.get("height")
        if isinstance(height_value, int):
            explicit_height = _clamp_webapp_height(height_value)
    else:
        html_content = str(props or "").strip()

    hinted_height, html_content = _extract_webapp_height(html_content)
    height = explicit_height or hinted_height

    if not html_content:
        st.info("Web app component is empty.")
        return

    iframe_src = _build_webapp_iframe_src(html_content)
    if not iframe_src:
        st.info("Web app component is empty.")
        return

    _inject_webapp_css()
    st.iframe(iframe_src, height=height)


# ===================================================================
# Component Registry
# ===================================================================

_COMPONENT_REGISTRY: Dict[str, Callable] = {
    "info_card": _render_info_card,
    "data_table": _render_data_table,
    "metric": _render_metric,
    "chart": _render_chart,
    "webapp": _render_webapp,
    "button_group": _render_button_group,
    "progress": _render_progress,
    "alert": _render_alert,
    "columns": _render_columns,
    "json_viewer": _render_json_viewer,
    "link_cards": _render_link_cards,
    "steps": _render_steps,
    "tabs": _render_tabs,
    "callout": _render_callout,
}


# ===================================================================
# Auto-enhancement: detect markdown patterns → UI blocks
# ===================================================================

# Matches a bold-label: value line  e.g.  "**Users:** 1,234" or "**Users**: 1,234"
_BOLD_KV_LINE_RE = re.compile(
    r'^\s*[-*]?\s*\*\*(.+?)\*\*\s*[:\-–]?\s*(.+?)\s*$',
)


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
        m = _BOLD_KV_LINE_RE.match(line)
        if m:
            if not group:
                group_start = i
            group.append({"label": m.group(1).rstrip(":- –").strip(), "value": m.group(2).strip()})
        else:
            if group:
                _flush()
            result_lines.append(line)
    if group:
        _flush()

    return '\n'.join(result_lines)


def auto_enhance_content(text: str) -> str:
    """Convert common markdown patterns to ``ui:*`` blocks.

    Called when the Generative UI toggle is enabled but the model did not
    emit any native ``ui:*`` fenced blocks (typical for smaller models).
    Converts markdown tables → ``ui:data_table`` and bold key-value
    metric lines → ``ui:metric``.
    """
    text = _auto_convert_tables(text)
    text = _auto_convert_metrics(text)
    return text


# ===================================================================
# System prompt for Generative UI  (reference only – the canonical
# prompt used at runtime is in app/backend/api/chat.py)
# ===================================================================

GENERATIVE_UI_SYSTEM_PROMPT = (
    "You can embed rich UI components in your responses using fenced code "
    "blocks with a ui: language tag followed by a JSON body on the NEXT line.\n\n"
    "Syntax: ```ui:component_type\\n{JSON props}\\n```\n\n"
    "Components:\n"
    '- data_table: {"title": "...", "columns": ["A","B"], "data": [["1","2"]]}\n'
    '- metric: {"metrics": [{"label": "...", "value": "...", "delta": "+5%"}]}\n'
    '- chart: {"type": "bar|line|pie|scatter", "title": "...", "x": [...], "y": [...]}\n'
    '- webapp: raw HTML/CSS/JS (not JSON), optionally starting with <!-- height: 640 -->\n'
    '- info_card: {"icon": "📊", "title": "...", "content": "...", "color": "#hex"}\n'
    '- button_group: {"label": "...", "buttons": [{"text": "...", "value": "user prompt"}]}\n'
    '- alert: {"level": "info|success|warning|error", "message": "..."}\n'
    '- steps: {"steps": ["Step1", "Step2"], "current": 1}\n'
    '- tabs: {"tabs": [{"label": "...", "content": "markdown text"}]}\n'
    '- callout: {"emoji": "💡", "title": "...", "content": "..."}\n'
    '- columns: {"items": [{"title": "...", "content": "...", "icon": "🔹"}]}\n'
    '- json_viewer: {"title": "...", "data": {...}}\n'
    '- progress: {"value": 75, "label": "..."}\n'
    '- link_cards: {"links": [{"title": "...", "url": "https://...", "description": "..."}]}\n\n'
    "Rules: Use valid JSON in component blocks. Combine text with components. "
    "If unsure about syntax, use standard markdown tables and "
    "**Bold Label:** Value lines – they will be auto-converted."
)
