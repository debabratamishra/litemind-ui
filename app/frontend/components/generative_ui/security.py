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
