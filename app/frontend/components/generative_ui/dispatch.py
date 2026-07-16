"""Component registry and top-level dispatcher for generative UI rendering."""

import json
import logging
from typing import Any, Callable, Dict

import streamlit as st

from .constants import _RAW_HTML_COMPONENT_TYPES
from .parsing import _is_valid_ui_component_type, _iter_fenced_blocks
from .renderers import (
    _render_alert,
    _render_button_group,
    _render_callout,
    _render_chart,
    _render_columns,
    _render_data_table,
    _render_info_card,
    _render_json_viewer,
    _render_link_cards,
    _render_metric,
    _render_progress,
    _render_steps,
    _render_tabs,
)
from .webapp import _render_iframe_app, _render_webapp

logger = logging.getLogger(__name__)


_COMPONENT_REGISTRY: Dict[str, Callable] = {
    "info_card": _render_info_card,
    "data_table": _render_data_table,
    "metric": _render_metric,
    "chart": _render_chart,
    "webapp": _render_webapp,
    "iframe_app": _render_iframe_app,
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
    # Three dots: dispatch.py lives one package level deeper than the old flat
    # generative_ui.py module did, so reaching app.frontend.utils requires
    # climbing past app.frontend.components.generative_ui -> components -> frontend.
    from ...utils.text_processing import (
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
                if component_type in _RAW_HTML_COMPONENT_TYPES:
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


def has_ui_blocks(text: str) -> bool:
    """Return *True* if *text* contains at least one complete UI block."""
    for _, _, lang, _ in _iter_fenced_blocks(text):
        if lang.startswith("ui:") and _is_valid_ui_component_type(lang[3:]):
            return True
    return False
