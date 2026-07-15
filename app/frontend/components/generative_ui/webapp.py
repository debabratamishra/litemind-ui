"""Web-app and iframe-app rendering, wrapping, and payload normalization."""

import base64
import json
import re
import streamlit as st
from typing import Any

from .constants import (
    _RAW_HTML_COMPONENT_TYPES,
    _WEBAPP_CSS,
    _WEBAPP_HEIGHT_RE,
    _IFRAME_APP_SHELL_CSS,
    _IFRAME_APP_BOOTSTRAP_SCRIPT,
    _INTERACTIVE_HTML_RE,
    _HTML_HEAD_CLOSE_RE,
    _HTML_BODY_CLOSE_RE,
)
from .security import _sanitize_html
from .parsing import (
    _extract_fenced_body,
    _try_parse_json_object,
    _extract_primary_html_document,
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
        '  <meta charset="utf-8" />\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1" />\n'
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


def _inject_html_fragment(document: str, pattern: re.Pattern[str], fragment: str) -> str:
    match = pattern.search(document)
    if match is None:
        return document + "\n" + fragment

    return document[:match.start()] + fragment + "\n" + document[match.start():]


def _compose_iframe_app_markup(props: dict) -> str:
    html_content = str(props.get("html", "")).strip()
    css_content = str(props.get("css", "")).strip()
    js_content = str(props.get("js", "")).strip()

    if not html_content and not css_content and not js_content:
        return ""

    fragments = [html_content] if html_content else ["<div id=\"app\" data-lite-app-root></div>"]
    if css_content:
        fragments.append(f"<style>\n{css_content}\n</style>")
    if js_content:
        fragments.append(f"<script>\n{js_content}\n</script>")
    return "\n".join(fragment for fragment in fragments if fragment)


def _component_type_for_html(markup: str) -> str:
    """Choose the richer iframe renderer when markup looks interactive."""
    return "iframe_app" if _INTERACTIVE_HTML_RE.search(markup) else "webapp"


def _normalise_webapp_payload(raw_payload: Any) -> str:
    """Normalise webapp payload across raw HTML and accidental JSON envelopes."""
    if isinstance(raw_payload, dict):
        html_content = str(raw_payload.get("html", "")).strip()
        if html_content:
            return html_content
        return json.dumps(raw_payload, ensure_ascii=False, indent=2)

    stripped = _extract_fenced_body(str(raw_payload or ""))
    if not stripped:
        return ""

    parsed = _try_parse_json_object(stripped)
    if parsed is not None:
        if "html" in parsed:
            return str(parsed.get("html", "")).strip()
        return json.dumps(parsed, ensure_ascii=False, indent=2)

    document = _extract_primary_html_document(stripped)
    if document is not None:
        return document

    return stripped


def _normalise_iframe_app_payload(raw_payload: Any) -> str:
    """Normalise iframe-app payload across dict, JSON, and fenced variants."""
    if isinstance(raw_payload, dict):
        return _compose_iframe_app_markup(raw_payload)

    stripped = _extract_fenced_body(str(raw_payload or ""))
    if not stripped:
        return ""

    parsed = _try_parse_json_object(stripped)
    if parsed is not None:
        if any(key in parsed for key in ("html", "css", "js")):
            return _compose_iframe_app_markup(parsed)
        return json.dumps(parsed, ensure_ascii=False, indent=2)

    document = _extract_primary_html_document(stripped)
    if document is not None:
        return document

    return stripped


def _wrap_iframe_app_html(html_content: str) -> str:
    stripped = html_content.strip()
    if not stripped:
        return ""

    if re.search(r"<html[\s>]", stripped, re.IGNORECASE):
        document = stripped
    elif re.search(r"<(head|body)[\s>]", stripped, re.IGNORECASE):
        document = (
            "<!DOCTYPE html>\n"
            "<html>\n"
            f"{stripped}\n"
            "</html>"
        )
    else:
        return (
            "<!DOCTYPE html>\n"
            "<html>\n"
            "<head>\n"
            '  <meta charset="utf-8" />\n'
            '  <meta name="viewport" content="width=device-width, initial-scale=1" />\n'
            f"{_IFRAME_APP_SHELL_CSS}\n"
            "</head>\n"
            "<body>\n"
            '  <div id="app" data-lite-app-root>\n'
            f"{stripped}\n"
            "  </div>\n"
            f"{_IFRAME_APP_BOOTSTRAP_SCRIPT}\n"
            "</body>\n"
            "</html>"
        )

    document = _inject_html_fragment(document, _HTML_HEAD_CLOSE_RE, _IFRAME_APP_SHELL_CSS)
    return _inject_html_fragment(document, _HTML_BODY_CLOSE_RE, _IFRAME_APP_BOOTSTRAP_SCRIPT)


def _build_iframe_app_iframe_src(html_content: str) -> str:
    wrapped_html = _wrap_iframe_app_html(html_content)
    if not wrapped_html:
        return ""

    encoded_html = base64.b64encode(wrapped_html.encode("utf-8")).decode("ascii")
    return f"data:text/html;base64,{encoded_html}"


def _build_webapp_iframe_src(html_content: str) -> str:
    wrapped_html = _wrap_webapp_html(html_content)
    if not wrapped_html:
        return ""

    encoded_html = base64.b64encode(wrapped_html.encode("utf-8")).decode("ascii")
    return f"data:text/html;base64,{encoded_html}"


def _render_webapp(props: Any, msg_index: int = 0) -> None:
    explicit_height = None
    if isinstance(props, dict):
        html_content = _normalise_webapp_payload(props)
        height_value = props.get("height")
        if isinstance(height_value, int):
            explicit_height = _clamp_webapp_height(height_value)
    else:
        html_content = _normalise_webapp_payload(props)

    hinted_height, html_content = _extract_webapp_height(html_content)
    height = explicit_height or hinted_height

    if not html_content:
        st.info("Web app component is empty.")
        return

    wrapped_html = _wrap_webapp_html(html_content)
    if not wrapped_html:
        st.info("Web app component is empty.")
        return

    _inject_webapp_css()
    # st.iframe() accepts raw HTML strings for inline app rendering.
    st.iframe(wrapped_html, height=height, tab_index=0)


def _render_iframe_app(props: Any, msg_index: int = 0) -> None:
    explicit_height = None
    title = ""
    description = ""

    if isinstance(props, dict):
        html_content = _normalise_iframe_app_payload(props)
        title = str(props.get("title", "")).strip()
        description = str(props.get("description", "")).strip()
        height_value = props.get("height")
        if isinstance(height_value, int):
            explicit_height = _clamp_webapp_height(height_value)
    else:
        html_content = _normalise_iframe_app_payload(props)

    hinted_height, html_content = _extract_webapp_height(html_content, default_height=720)
    height = explicit_height or hinted_height

    if title:
        st.markdown(f"**{_sanitize_html(title)}**")
    if description:
        st.caption(description)

    if not html_content:
        st.info("Iframe app component is empty.")
        return

    wrapped_html = _wrap_iframe_app_html(html_content)
    if not wrapped_html:
        st.info("Iframe app component is empty.")
        return

    _inject_webapp_css()
    st.caption("Click inside the frame to interact · keyboard arrow keys and mouse both work.")
    # st.iframe() accepts raw HTML strings for inline app rendering.
    st.iframe(wrapped_html, height=height, tab_index=0)
