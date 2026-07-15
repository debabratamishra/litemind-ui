# Decompose `generative_ui.py` into a package — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the 1,429-line `app/frontend/components/generative_ui.py` into a `generative_ui/` package of cohesive modules with identical behavior and an unchanged public API.

**Architecture:** Pure verbatim relocation of functions into leaf-to-root modules (`constants`, `security`, `parsing` → `renderers`, `webapp`, `auto_enhance` → `dispatch` + `__init__`). Acyclic imports; the package `__init__.py` re-exports `has_ui_blocks`, `render_mixed_content`, `auto_enhance_content` so `text_renderer.py` needs no edits.

**Tech Stack:** Python 3 (existing Streamlit/FastAPI codebase), `uv` for running tools (`uv run ruff check .`, `uv run python -c ...`).

## Global Constraints

- Behavior, function signatures, and rendered output must be **identical** — this is a relocation only, no logic/naming changes.
- The package must re-export `has_ui_blocks`, `render_mixed_content`, `auto_enhance_content` (and `render_ui_component`, `GENERATIVE_UI_SYSTEM_PROMPT`) so `text_renderer.py` is untouched.
- No new tests (user decision). Verification is via `uv run ruff check .` and import checks.
- Commit frequently, one module per commit.
- `render_mixed_content` keeps its lazy `from ..utils.text_processing import (clean_markdown_text, sanitize_links, unescape_text)`.

---

## File Structure

```
app/frontend/components/generative_ui/        # NEW package (replaces generative_ui.py)
├── __init__.py        # module docstring + re-exports public API
├── constants.py       # LEAF: _RAW_HTML_COMPONENT_TYPES, CSS/JS strings, regexes, system prompt
├── security.py        # LEAF: _sanitize_html, _sanitize_url
├── parsing.py         # LEAF: fence + HTML-document parsing, JSON helpers (no streamlit)
├── renderers.py       # 13 simple _render_* components  → security
├── webapp.py          # webapp/iframe rendering + normalization  → parsing, security, constants
├── auto_enhance.py    # markdown→ui conversions  → parsing, webapp
└── dispatch.py        # _COMPONENT_REGISTRY, render_ui_component, render_mixed_content, has_ui_blocks  → parsing, constants, renderers, webapp

Delete: app/frontend/components/generative_ui.py
```

---

### Task 1: Create `constants.py`

**Files:**
- Create: `app/frontend/components/generative_ui/constants.py`

**Interfaces:**
- Produces (imported by later tasks): `_RAW_HTML_COMPONENT_TYPES`, `_WEBAPP_HEIGHT_RE`, `_INTERACTIVE_HTML_RE`, `_WEBAPP_CSS`, `_HTML_HEAD_CLOSE_RE`, `_HTML_BODY_CLOSE_RE`, `_IFRAME_APP_SHELL_CSS`, `_IFRAME_APP_BOOTSTRAP_SCRIPT`, `GENERATIVE_UI_SYSTEM_PROMPT`.

- [ ] **Step 1: Write the module**

```python
"""Shared constants for generative UI rendering (CSS shells, JS bootstrap, regexes, system prompt)."""

import re

_RAW_HTML_COMPONENT_TYPES = {"webapp", "iframe_app"}

_WEBAPP_HEIGHT_RE = re.compile(r"^\s*<!--\s*height\s*:\s*(\d{2,4})\s*-->\s*", re.IGNORECASE)
_INTERACTIVE_HTML_RE = re.compile(
    r"(?is)<(?:script|canvas|button|input|select|textarea)\b|"
    r"on(?:click|change|input|submit|keydown|keyup|pointerdown)\s*=|"
    r"addEventListener\s*\(|requestAnimationFrame\s*\("
)

_WEBAPP_CSS = """
<style>
/* Allow the chat message content to pass pointer events through */
[data-testid="stChatMessageContent"] {
    pointer-events: auto !important;
}

/* Ensure every level of the component wrapper passes events through.
    Streamlit may wrap iframe blocks with intermediate containers depending on
    the element type, so target both direct-child and descendant iframes. */
[data-testid="stCustomComponentV1"],
[data-testid="stCustomComponentV1"] > div,
[data-testid="stCustomComponentV1"] > div > iframe,
[data-testid="stCustomComponentV1"] iframe,
[data-testid="stIFrame"],
[data-testid="stIFrame"] > iframe,
.stCustomComponentV1,
.stCustomComponentV1 iframe {
    pointer-events: auto !important;
}

/* Remove any pseudo-element overlays Streamlit adds on chat bubbles */
[data-testid="stChatMessage"]::before,
[data-testid="stChatMessage"]::after,
[data-testid="stChatMessageContent"]::before,
[data-testid="stChatMessageContent"]::after {
    pointer-events: none !important;
}

/* Hide toolbar overlays that sit above component iframes */
div[data-testid="stElementToolbar"],
div[data-testid="stElementToolbar"] * {
    pointer-events: none !important;
    display: none !important;
}
</style>
"""

_HTML_HEAD_CLOSE_RE = re.compile(r"</head\s*>", re.IGNORECASE)
_HTML_BODY_CLOSE_RE = re.compile(r"</body\s*>", re.IGNORECASE)

_IFRAME_APP_SHELL_CSS = """
<style>
    html, body {
        margin: 0;
        padding: 0;
        width: 100%;
        min-height: 100%;
    }

    body {
        font-family: "SF Pro Text", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: #ffffff;
        color: #111827;
        overflow: auto;
    }

    *, *::before, *::after {
        box-sizing: border-box;
    }

    #app[data-lite-app-root] {
        min-height: 100vh;
        width: 100%;
    }

    canvas {
        display: block;
        max-width: 100%;
        touch-action: none;
    }

    button,
    input,
    select,
    textarea {
        font: inherit;
    }
 </style>
"""

_IFRAME_APP_BOOTSTRAP_SCRIPT = """
<script>
(function () {
    "use strict";

    var BLOCKED_KEYS = new Set(["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", " "]);

    // ------------------------------------------------------------------
    // Utilities
    // ------------------------------------------------------------------
    function isEditable(el) {
        if (!el) return false;
        var tag = (el.tagName || "").toUpperCase();
        return el.isContentEditable || tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
    }

    function showRuntimeError(message) {
        var text = String(message || "Unknown iframe app error");
        if (!text || text === "Script error.") {
            // Browsers mask cross-origin script exceptions as "Script error."
            // This is usually not actionable for end users.
            return;
        }

        var banner = document.getElementById("litemind-app-error");
        if (!banner) {
            banner = document.createElement("div");
            banner.id = "litemind-app-error";
            banner.style.cssText = [
                "position:fixed", "left:12px", "right:12px", "bottom:12px",
                "padding:10px 12px", "border-radius:12px",
                "background:rgba(127,29,29,0.94)", "color:#fff",
                'font:500 12px/1.4 "SF Pro Text",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif',
                "box-shadow:0 12px 30px rgba(15,23,42,.28)",
                "z-index:2147483647", "white-space:pre-wrap"
            ].join(";");
            document.body.appendChild(banner);
        }
        banner.textContent = text;
    }

    // ------------------------------------------------------------------
    // Parent-page fixes
    //
    // If the iframe runs same-origin with the parent Streamlit page, we can
    // access window.parent.document to:
    //   1. Focus the <iframe> element itself in the parent DOM — this is
    //      what routes keyboard events to us without requiring a prior
    //      user click.
    //   2. Install a capture-phase keydown handler on the parent that
    //      prevents arrow-key page scroll only while an iframe is the
    //      active element (so normal Streamlit page scroll is preserved).
    //   3. Hide the element toolbar that Streamlit floats over each
    //      component on hover (it blocks the first mouse click otherwise).
    // ------------------------------------------------------------------
    (function installParentFixes() {
        var parentWin, parentDoc;
        try {
            parentWin = window.parent;
            parentDoc = parentWin.document;
            void parentDoc.body; // throws if cross-origin
        } catch (e) {
            return; // not same-origin — skip
        }

        // 1. Arrow-key scroll prevention on the parent Streamlit page.
        //    Only fires when an <iframe> element is the active element,
        //    so regular page scrolling is unaffected.
        if (!parentWin._litemindArrowKeyFixInstalled) {
            parentWin._litemindArrowKeyFixInstalled = true;
            parentDoc.addEventListener("keydown", function (e) {
                if (BLOCKED_KEYS.has(e.key)) {
                    var active = parentDoc.activeElement;
                    if (active && active.tagName === "IFRAME") {
                        e.preventDefault();
                    }
                }
            }, { capture: true, passive: false });
        }

        // 2. Find the <iframe> element in the parent that wraps THIS window.
        function findSelfIframe() {
            var frames = parentDoc.querySelectorAll("iframe");
            for (var i = 0; i < frames.length; i++) {
                try {
                    if (frames[i].contentWindow === window) return frames[i];
                } catch (err) { /* skip any cross-origin sibling */ }
            }
            return null;
        }

        // Focus the <iframe> element in the parent so keyboard events are
        // routed here without requiring the user to click first.
        function focusSelf() {
            var el = findSelfIframe();
            if (!el) return;
            if (!el.getAttribute("tabindex")) el.setAttribute("tabindex", "0");
            el.focus({ preventScroll: true });
        }

        // Hide the Streamlit element toolbar that floats above this component.
        // We target only the toolbar that is a DOM sibling of our wrapper,
        // leaving toolbars on other elements untouched.
        function hideNearbyToolbar() {
            var el = findSelfIframe();
            if (!el) return;
            var wrapper = el.closest
                ? el.closest('[data-testid="stIFrame"], [data-testid="stCustomComponentV1"]')
                : null;
            var container = wrapper ? wrapper.parentElement : null;
            if (!container) return;
            var toolbar = container.querySelector('[data-testid="stElementToolbar"]');
            if (toolbar) {
                toolbar.style.setProperty("display",        "none",  "important");
                toolbar.style.setProperty("pointer-events", "none",  "important");
            }
        }

        window.addEventListener("load", function () {
            requestAnimationFrame(function () {
                hideNearbyToolbar();
                focusSelf();
            });
        });

        // Re-focus every time the user clicks/taps inside the app.
        document.addEventListener("pointerdown", function () {
            requestAnimationFrame(focusSelf);
        }, { passive: true });

        // Export so LiteMindApp.requestFocus() works from game code.
        window._litemindFocusSelf = focusSelf;
    })();

    // ------------------------------------------------------------------
    // Inside-iframe: also prevent arrow keys from scrolling the iframe
    // document itself (belt-and-suspenders).
    // ------------------------------------------------------------------
    window.addEventListener("keydown", function (e) {
        if (BLOCKED_KEYS.has(e.key) && !isEditable(e.target)) {
            e.preventDefault();
        }
    }, { capture: true });

    // ------------------------------------------------------------------
    // Error reporting
    // ------------------------------------------------------------------
    window.addEventListener("error", function (e) {
        showRuntimeError(e.message || "Iframe app error");
    });

    window.addEventListener("unhandledrejection", function (e) {
        var r = e.reason;
        if (!r) {
            return;
        }
        showRuntimeError(r && typeof r === "object" && r.message
            ? r.message
            : String(r || "Unhandled promise rejection"));
    });

    // ------------------------------------------------------------------
    // LiteMind app API
    // ------------------------------------------------------------------
    window.LiteMindApp = Object.assign({}, window.LiteMindApp || {}, {
        requestFocus: function () {
            if (window._litemindFocusSelf) {
                window._litemindFocusSelf();
            }
        },
        showRuntimeError: showRuntimeError,
        getViewport: function () {
            return { width: window.innerWidth, height: window.innerHeight };
        }
    });
})();
</script>
"""

GENERATIVE_UI_SYSTEM_PROMPT = (
    "You can embed rich UI components in your responses using fenced code "
    "blocks with a ui: language tag followed by a JSON body on the NEXT line.\n\n"
    "Syntax: ```ui:component_type\\n{JSON props}\\n```\n\n"
    "Components:\n"
    '- data_table: {"title": "...", "columns": ["A","B"], "data": [["1","2"]]}\n'
    '- metric: {"metrics": [{"label": "...", "value": "...", "delta": "+5%"}]}\n'
    '- chart: {"type": "bar|line|pie|scatter", "title": "...", "x": [...], "y": [...]}\n'
    '- webapp: raw HTML/CSS/JS (not JSON), optionally starting with <!-- height: 640 -->\n'
    '- iframe_app: raw HTML/CSS/JS for playable apps and games, optionally starting with <!-- height: 720 -->\n'
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
    "Rules: Use valid JSON in component blocks. Use iframe_app for playable apps and games. Combine text with components. "
    "If unsure about syntax, use standard markdown tables and "
    "**Bold Label:** Value lines – they will be auto-converted."
)
```

- [ ] **Step 2: Verify the module imports in isolation**

Run: `uv run python -c "import app.frontend.components.generative_ui.constants as c; print(c.GENERATIVE_UI_SYSTEM_PROMPT[:20])"`
Expected: prints the first 20 chars of the prompt (no import error).

- [ ] **Step 3: Commit**

```bash
git add app/frontend/components/generative_ui/constants.py
git commit -m "refactor: extract generative_ui constants module"
```

---

### Task 2: Create `security.py`

**Files:**
- Create: `app/frontend/components/generative_ui/security.py`

**Interfaces:**
- Produces: `_sanitize_html`, `_sanitize_url` (consumed by `renderers.py` and `webapp.py`).

- [ ] **Step 1: Write the module**

```python
"""HTML/URL sanitizers for untrusted LLM-provided content."""

import html


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
```

- [ ] **Step 2: Verify the module imports in isolation**

Run: `uv run python -c "from app.frontend.components.generative_ui.security import _sanitize_html, _sanitize_url; print(_sanitize_url('javascript:alert(1)'))"`
Expected: prints `#`

- [ ] **Step 3: Commit**

```bash
git add app/frontend/components/generative_ui/security.py
git commit -m "refactor: extract generative_ui security helpers"
```

---

### Task 3: Create `parsing.py`

**Files:**
- Create: `app/frontend/components/generative_ui/parsing.py`

**Interfaces:**
- Produces: `_is_valid_fence_info`, `_is_valid_ui_component_type`, `_iter_fenced_blocks`, `_extract_fenced_body`, `_match_single_fenced_block`, `_find_html_code_fence`, `_try_parse_json_object`, `_is_html_tag_boundary`, `_parse_doctype_html_tag_end`, `_parse_html_open_tag_end`, `_parse_html_close_tag_end`, `_find_primary_html_document_span`, `_extract_primary_html_document`.
- Depends only on stdlib (`json`). No Streamlit. Consumed by `webapp.py`, `auto_enhance.py`, `dispatch.py`.

- [ ] **Step 1: Write the module**

```python
"""Fenced-block and embedded-HTML parsing helpers (pure, no Streamlit)."""

import json


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
```

Note: add `from typing import Iterator, Optional` at the top of the file (the `Iterator`/`Optional` annotations above require it). The completed module header is:

```python
"""Fenced-block and embedded-HTML parsing helpers (pure, no Streamlit)."""

import json
from typing import Iterator, Optional
```

- [ ] **Step 2: Verify the module imports in isolation**

Run: `uv run python -c "from app.frontend.components.generative_ui.parsing import _iter_fenced_blocks, _extract_primary_html_document; print('ok')"`
Expected: prints `ok`

- [ ] **Step 3: Commit**

```bash
git add app/frontend/components/generative_ui/parsing.py
git commit -m "refactor: extract generative_ui parsing helpers"
```

---

### Task 4: Create `renderers.py`

**Files:**
- Create: `app/frontend/components/generative_ui/renderers.py`

**Interfaces:**
- Consumes: `from .security import _sanitize_html, _sanitize_url`.
- Produces: `_render_info_card`, `_render_data_table`, `_render_metric`, `_render_chart`, `_render_button_group`, `_render_progress`, `_render_alert`, `_render_columns`, `_render_json_viewer`, `_render_link_cards`, `_render_steps`, `_render_tabs`, `_render_callout`. All consumed by `dispatch.py`'s `_COMPONENT_REGISTRY`.

- [ ] **Step 1: Write the module**

```python
"""Simple Streamlit component renderers for generative UI blocks."""

import streamlit as st

from .security import _sanitize_html, _sanitize_url


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
```

- [ ] **Step 2: Verify the module imports (security must exist from Task 2)**

Run: `uv run python -c "from app.frontend.components.generative_ui.renderers import _render_info_card, _render_data_table; print('ok')"`
Expected: prints `ok`

- [ ] **Step 3: Commit**

```bash
git add app/frontend/components/generative_ui/renderers.py
git commit -m "refactor: extract generative_ui simple renderers"
```

---

### Task 5: Create `webapp.py`

**Files:**
- Create: `app/frontend/components/generative_ui/webapp.py`

**Interfaces:**
- Consumes: `from .constants import (_RAW_HTML_COMPONENT_TYPES, _WEBAPP_CSS, _WEBAPP_HEIGHT_RE, _IFRAME_APP_SHELL_CSS, _IFRAME_APP_BOOTSTRAP_SCRIPT, _INTERACTIVE_HTML_RE, _HTML_HEAD_CLOSE_RE, _HTML_BODY_CLOSE_RE)`, `from .security import _sanitize_html`, `from .parsing import (_extract_fenced_body, _try_parse_json_object, _extract_primary_html_document)`.
- Produces: `_inject_webapp_css`, `_clamp_webapp_height`, `_extract_webapp_height`, `_wrap_webapp_html`, `_wrap_iframe_app_html`, `_inject_html_fragment`, `_compose_iframe_app_markup`, `_component_type_for_html`, `_normalise_webapp_payload`, `_normalise_iframe_app_payload`, `_build_iframe_app_iframe_src`, `_build_webapp_iframe_src`, `_render_webapp`, `_render_iframe_app`. Consumed by `dispatch.py` (`_render_webapp`, `_render_iframe_app`) and `auto_enhance.py` (`_component_type_for_html`).

- [ ] **Step 1: Write the module**

```python
"""Web-app and iframe-app rendering, wrapping, and payload normalization."""

import base64
import json
import re
import streamlit as st

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
            "  <meta charset=\"utf-8\" />\n"
            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />\n"
            f"{_IFRAME_APP_SHELL_CSS}\n"
            "</head>\n"
            "<body>\n"
            "  <div id=\"app\" data-lite-app-root>\n"
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
```

Add `from typing import Any` at the top (used by `_normalise_webapp_payload`, `_render_webapp`, `_render_iframe_app`). Final module header:

```python
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
```

- [ ] **Step 2: Verify the module imports (depends on Tasks 1–3)**

Run: `uv run python -c "from app.frontend.components.generative_ui.webapp import _render_webapp, _component_type_for_html; print('ok')"`
Expected: prints `ok`

- [ ] **Step 3: Commit**

```bash
git add app/frontend/components/generative_ui/webapp.py
git commit -m "refactor: extract generative_ui webapp/iframe rendering"
```

---

### Task 6: Create `auto_enhance.py`

**Files:**
- Create: `app/frontend/components/generative_ui/auto_enhance.py`

**Interfaces:**
- Consumes: `from .parsing import (_find_html_code_fence, _extract_primary_html_document_span)`, `from .webapp import _component_type_for_html`.
- Produces: `auto_enhance_content` (public), plus `_match_bold_kv_line`, `_split_md_row`, `_looks_like_md_table_row`, `_is_md_table_separator_cell`, `_is_md_table_separator_row`, `_parse_md_table`, `_auto_convert_tables`, `_auto_convert_metrics`, `_wrap_html_markup_as_ui_block`, `_auto_convert_html_markup`.

- [ ] **Step 1: Write the module**

```python
"""Auto-enhancement: detect markdown patterns and convert them to ui:* blocks."""

import json

from .parsing import (
    _find_html_code_fence,
    _extract_primary_html_document_span,
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

    document_span = _extract_primary_html_document_span(text)
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
```

Add `from typing import Optional` at the top (used by `_parse_md_table`). Final module header:

```python
"""Auto-enhancement: detect markdown patterns and convert them to ui:* blocks."""

import json
from typing import Optional

from .parsing import (
    _find_html_code_fence,
    _extract_primary_html_document_span,
)
from .webapp import _component_type_for_html
```

- [ ] **Step 2: Verify the module imports (depends on Tasks 3, 5)**

Run: `uv run python -c "from app.frontend.components.generative_ui.auto_enhance import auto_enhance_content; print(auto_enhance_content('**Users:** 1,234\\n**Active:** 42'))"`
Expected: prints text containing a `ui:metric` fenced block.

- [ ] **Step 3: Commit**

```bash
git add app/frontend/components/generative_ui/auto_enhance.py
git commit -m "refactor: extract generative_ui auto-enhancement"
```

---

### Task 7: Create `dispatch.py` + `__init__.py`, then delete old module

**Files:**
- Create: `app/frontend/components/generative_ui/dispatch.py`
- Create: `app/frontend/components/generative_ui/__init__.py`
- Delete: `app/frontend/components/generative_ui.py`

**Interfaces:**
- Consumes: `from .parsing import _iter_fenced_blocks, _is_valid_ui_component_type`, `from .constants import _RAW_HTML_COMPONENT_TYPES`, `from .renderers import (... 13 renderers ...)`, `from .webapp import _render_webapp, _render_iframe_app`.
- Produces (public API, re-exported from `__init__.py`): `has_ui_blocks`, `render_mixed_content`, `render_ui_component`, `auto_enhance_content` (re-exported from auto_enhance), `GENERATIVE_UI_SYSTEM_PROMPT` (from constants), `_COMPONENT_REGISTRY`.

- [ ] **Step 1: Write `dispatch.py`**

```python
"""Component registry and top-level dispatcher for generative UI rendering."""

import logging
import streamlit as st

from .parsing import _iter_fenced_blocks, _is_valid_ui_component_type
from .constants import _RAW_HTML_COMPONENT_TYPES
from .renderers import (
    _render_info_card,
    _render_data_table,
    _render_metric,
    _render_chart,
    _render_button_group,
    _render_progress,
    _render_alert,
    _render_columns,
    _render_json_viewer,
    _render_link_cards,
    _render_steps,
    _render_tabs,
    _render_callout,
)
from .webapp import _render_webapp, _render_iframe_app

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
```

Add `from typing import Any, Callable, Dict` at the top (used by `render_ui_component`, `_COMPONENT_REGISTRY`). Final module header:

```python
"""Component registry and top-level dispatcher for generative UI rendering."""

import json
import logging
import streamlit as st
from typing import Any, Callable, Dict

from .parsing import _iter_fenced_blocks, _is_valid_ui_component_type
from .constants import _RAW_HTML_COMPONENT_TYPES
from .renderers import (
    _render_info_card,
    _render_data_table,
    _render_metric,
    _render_chart,
    _render_button_group,
    _render_progress,
    _render_alert,
    _render_columns,
    _render_json_viewer,
    _render_link_cards,
    _render_steps,
    _render_tabs,
    _render_callout,
)
from .webapp import _render_webapp, _render_iframe_app

logger = logging.getLogger(__name__)
```

- [ ] **Step 2: Write `__init__.py` (carries the original module docstring and re-exports the public API)**

```python
"""
Generative UI components for dynamic rendering in chat responses.

Inspired by the AG-UI (Agent-User Interaction) protocol's event-driven
component model. Renders rich, interactive UI elements from structured
specifications embedded in LLM output.

Protocol: LLM emits fenced code blocks with a ``ui:<component>`` language
tag. Most components use a JSON props object. ``ui:webapp`` and
``ui:iframe_app`` are the exceptions: their block bodies are raw
HTML/CSS/JS rendered inside a Streamlit iframe for interactive mini-apps.
"""

from .dispatch import (
    has_ui_blocks,
    render_mixed_content,
    render_ui_component,
    _COMPONENT_REGISTRY,
)
from .auto_enhance import auto_enhance_content
from .constants import GENERATIVE_UI_SYSTEM_PROMPT

__all__ = [
    "has_ui_blocks",
    "render_mixed_content",
    "render_ui_component",
    "auto_enhance_content",
    "GENERATIVE_UI_SYSTEM_PROMPT",
    "_COMPONENT_REGISTRY",
]
```

- [ ] **Step 3: Delete the old module file**

Run: `git rm app/frontend/components/generative_ui.py`

- [ ] **Step 4: Verify the package imports and the public API resolves**

Run: `uv run python -c "from app.frontend.components.generative_ui import has_ui_blocks, render_mixed_content, auto_enhance_content, render_ui_component, GENERATIVE_UI_SYSTEM_PROMPT; print('api ok')"`
Expected: prints `api ok`

Also confirm the caller still imports:
Run: `uv run python -c "import app.frontend.components.text_renderer; print('caller ok')"`
Expected: prints `caller ok` (no ImportError).

- [ ] **Step 5: Commit**

```bash
git add app/frontend/components/generative_ui/dispatch.py \
        app/frontend/components/generative_ui/__init__.py \
        app/frontend/components/generative_ui.py
git commit -m "refactor: add generative_ui dispatch package and remove flat module"
```

---

### Task 8: Repo-wide verification

**Files:**
- None new. Final lint + import gate across the package.

- [ ] **Step 1: Run the linter**

Run: `uv run ruff check .`
Expected: no errors in `app/frontend/components/generative_ui/`.

- [ ] **Step 2: Run a full import smoke test**

Run: `uv run python -c "import app.frontend.components.generative_ui as g; print(g.has_ui_blocks('x'), g.has_ui_blocks('\`\`\`ui:metric\n{}\n\`\`\`'))"`
Expected: prints `False True`.

- [ ] **Step 3: Final commit if any fixups were needed**

```bash
git add -A
git commit -m "refactor: finalise generative_ui package lint fixes"  # only if fixes were required
```

---

## Self-Review Notes

- **Spec coverage:** All 58 functions/constants from the original file are placed in exactly one module (see design doc mapping). Public API re-exported. Old module deleted.
- **Type consistency:** `render_ui_component(component_type, props, msg_index)` signature matches its single call site in `render_mixed_content`. `_COMPONENT_REGISTRY` keys/order identical to original. `auto_enhance_content` flow (`_auto_convert_html_markup → _auto_convert_tables → _auto_convert_metrics`) preserved.
- **No placeholders:** Every task contains the complete module code; verification commands have expected output.
