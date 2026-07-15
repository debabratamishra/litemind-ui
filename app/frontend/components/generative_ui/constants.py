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
