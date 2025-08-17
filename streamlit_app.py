"""
Streamlit front-end for the LLM WebUI with multimodal RAG support.

This module provides:
- A chat interface backed by a FastAPI service or a local Ollama fallback.
- A RAG workflow with enhanced document processing (CSV/image-aware) and hybrid search.
The code emphasizes clear structure, type hints, and concise logging.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st

from app.services.ollama import stream_ollama

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FASTAPI_URL: str = "http://localhost:8000"
FASTAPI_TIMEOUT: int = 60  # seconds
CONNECT_TIMEOUT: int = 5   # seconds
READ_TIMEOUT: int = 600    # seconds


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _unescape_text(s: str) -> str:
    """Convert visible escape sequences (e.g., ``\n``, ``\t``, ``\u2713``) into
    their actual characters, while avoiding double-unescaping."""
    if not isinstance(s, str):
        s = str(s)

    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]

    try:
        s = json.loads(f'"{s}"')
    except Exception:
        s = s.replace('\\\n', '\n').replace('\\\t', '\t')
        s = s.replace('\n', '\n').replace('\t', '\t')

    return s.replace('\\r\\n', '\n').replace('\r\n', '\n')


# ---------------------------------------------------------------------------
# Markdown/URL sanitization helpers
# ---------------------------------------------------------------------------

def _looks_like_url(text: str) -> bool:
    """Heuristic: return True if the text appears to be a URL."""
    return bool(re.match(r"^\s*(https?://|www\.)", text, re.IGNORECASE))


def _clean_url(url: str) -> str:
    """Normalize a URL string that may contain stray spaces.
    - Fix spaced scheme separators like "http ://".
    - Remove internal spaces but preserve URL structure.
    - Prepend https:// for leading www.
    """
    u = url.strip()
    
    # Handle www. prefix
    if u.startswith("www."):
        u = "https://" + u
    
    # Normalize spaced scheme separators, keep lowercase scheme
    u = re.sub(r"(?i)(https?)\s*:\s*/\s*/", r"\1://", u)
    
    # Remove spaces around dots in domain names
    u = re.sub(r"\s*\.\s*", ".", u)
    
    # Remove spaces around slashes
    u = re.sub(r"\s*/\s*", "/", u)
    
    # Remove spaces around colons (simple approach)
    u = re.sub(r"\s*:\s*", ":", u)
    # But fix the protocol part
    u = re.sub(r"(https?):(//)", r"\1://", u)
    
    # Remove internal spaces in the URL
    # Split by spaces and rejoin non-space parts
    parts = u.split()
    if len(parts) > 1:
        # If there are spaces, the first part should be the URL
        clean_url = parts[0]
        # Remove any remaining internal spaces in the URL part
        clean_url = re.sub(r'\s+', '', clean_url)
        return clean_url
    else:
        # No spaces, just clean any remaining whitespace
        return re.sub(r'\s+', '', u)


def sanitize_links(text: str) -> str:
    """Fix common hyperlink formatting issues in Markdown text.
    1) Clean URLs inside Markdown links: [text](url)
    2) Normalize bare http/https URLs that have embedded spaces.
    Does not modify fenced code blocks (handled by caller splitting blocks).
    """

    # 1) Clean URLs inside Markdown links
    def _md_link_repl(m: re.Match) -> str:
        disp, url = m.group(1), m.group(2)
        clean_url = _clean_url(url)
        clean_disp = _clean_url(disp) if _looks_like_url(disp) else disp
        return f"[{clean_disp}]({clean_url})"

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _md_link_repl, text)

    # 2) Normalize bare URLs with better boundary detection
    # Match URLs more precisely to avoid capturing following text
    def _bare_url_repl(m: re.Match) -> str:
        url_part = m.group(0)
        # Clean the URL but preserve any trailing punctuation that's not part of URL
        cleaned = _clean_url(url_part)
        return cleaned

    # More comprehensive pattern to catch URLs with spaces
    # This pattern specifically looks for URLs that may have spaces in them
    bare_url_pattern = r"(?i)https?\s*:\s*/\s*/(?:[a-zA-Z0-9\-._~:/?#[\]@!$&'()*+,;=%]|\s)+?(?=\s+[A-Z][a-z]|\s*[.!?]\s|$|\s+\w+:)"
    
    # Fallback pattern for simpler URLs
    simple_url_pattern = r"(?i)https?\s*:\s*/\s*/[^\s<>\"\'\(\)\[\]{}|\\^`]+(?:[^\s<>\"\'\(\)\[\]{}|\\^`.,;!?]|[.,;!?](?=\s|$))*"
    
    # Apply both patterns
    text = re.sub(bare_url_pattern, _bare_url_repl, text)
    text = re.sub(simple_url_pattern, _bare_url_repl, text)
    return text


def _extract_thinking_content(text: str) -> tuple[str, str]:
    """Extract thinking/reasoning content from text and return (thinking, clean_answer)."""
    
    # Pattern to match various thinking tag formats
    thinking_patterns = [
        r'<think>(.*?)</think>',           # <think>...</think>
        r'<thinking>(.*?)</thinking>',     # <thinking>...</thinking>
        r'<reasoning>(.*?)</reasoning>',   # <reasoning>...</reasoning>
        r'<thought>(.*?)</thought>',       # <thought>...</thought>
    ]
    
    thinking_content = []
    clean_text = text
    
    for pattern in thinking_patterns:
        matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            thinking_content.append(match.group(1).strip())
            # Remove the thinking tags from the clean text
            clean_text = re.sub(pattern, '', clean_text, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up any extra whitespace left after removing thinking tags
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text).strip()
    
    # Combine all thinking content
    combined_thinking = '\n\n'.join(thinking_content) if thinking_content else ''
    
    return combined_thinking, clean_text


def _render_thinking_section(thinking_content: str) -> None:
    """Render the thinking/reasoning section in a collapsible expander."""
    if not thinking_content.strip():
        return
    
    # Check if user wants to see reasoning by default (can be configured)
    default_expanded = st.session_state.get("show_reasoning_expanded", False)
    
    with st.expander("🧠 Model Reasoning Process", expanded=default_expanded):
        st.markdown("*This section shows the model's internal reasoning and thought process:*")
        st.markdown("---")
        
        # Clean and format the thinking content
        cleaned_thinking = _clean_text_formatting(thinking_content)
        cleaned_thinking = _clean_markdown_text(cleaned_thinking)
        
        # Render with a styled container
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, rgba(63, 81, 181, 0.1), rgba(33, 150, 243, 0.05));
                padding: 1.2rem;
                border-radius: 0.75rem;
                border-left: 4px solid #3f51b5;
                margin: 0.5rem 0;
                font-family: 'SF Pro Text', -apple-system, BlinkMacSystemFont, sans-serif;
            ">
                {sanitize_links(cleaned_thinking)}
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Add a small note about the reasoning
        st.caption("💡 This reasoning helps understand how the model arrived at its answer.")


def render_llm_text(s: str) -> None:
    """Render mixed Markdown and fenced code blocks in Streamlit with reasoning support."""
    text = _unescape_text(s).strip()
    
    # Extract thinking/reasoning content first
    thinking_content, clean_text = _extract_thinking_content(text)
    
    # Clean up common formatting issues
    clean_text = _clean_text_formatting(clean_text)
    
    # Render the thinking section if present and not hidden
    hide_reasoning = st.session_state.get("hide_reasoning", False)
    if thinking_content and not hide_reasoning:
        _render_thinking_section(thinking_content)
        st.markdown("---")  # Separator between thinking and answer
    
    # Render the main answer content
    code_fence = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)

    pos = 0
    last_end = 0
    rendered_any = False

    for match in code_fence.finditer(clean_text):
        before = clean_text[pos:match.start()]
        if before.strip():
            # Clean and render the text before code block
            cleaned_before = _clean_markdown_text(before)
            st.markdown(sanitize_links(cleaned_before))
            rendered_any = True

        lang = (match.group(1) or "").strip()
        code = match.group(2).strip("\n")
        st.code(code, language=lang if lang else None)
        rendered_any = True

        pos = match.end()
        last_end = match.end()

    tail = clean_text[last_end:] if last_end else clean_text
    if tail.strip() or not rendered_any:
        cleaned_tail = _clean_markdown_text(tail)
        st.markdown(sanitize_links(cleaned_tail))


def _clean_text_formatting(text: str) -> str:
    """Clean up common text formatting issues while preserving word spacing."""
    # Fix concatenated text after URLs first - this is the key fix
    text = _fix_concatenated_text_after_urls(text)
    
    # Fix numbered list formatting - ensure proper line breaks before numbers
    text = _fix_numbered_lists(text)
    
    # Normalize excessive whitespace but preserve single spaces between words
    text = re.sub(r'[ \t]{2,}', ' ', text)  # Only collapse multiple spaces/tabs
    
    # Fix broken headers (headers that appear mid-sentence)
    text = re.sub(r'([a-z])\s*\n\s*(#{1,6}\s+[A-Z])', r'\1\n\n\2', text)
    
    # Normalize line breaks - max 2 consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Fix spacing around punctuation (but be more careful)
    text = re.sub(r'\s{2,}([.!?,:;])', r' \1', text)  # Multiple spaces before punctuation
    text = re.sub(r'([.!?])\s*\n\s*([a-z])', r'\1 \2', text)  # Join broken sentences
    
    # Ensure proper spacing after URLs and before following text
    text = re.sub(r'(https?://[^\s]+)([A-Z][a-z])', r'\1 \2', text)
    
    return text


def _fix_numbered_lists(text: str) -> str:
    """Fix numbered list formatting by ensuring proper line breaks."""
    
    # Pattern to detect numbered list items that are inline instead of on new lines
    # Look for: sentence ending + space + number + period + space + capital letter
    numbered_list_pattern = r'([.!?])\s+(\d+\.\s+)([A-Z])'
    
    # Replace with proper line breaks
    text = re.sub(numbered_list_pattern, r'\1\n\n\2\3', text)
    
    # Also handle cases where numbered items follow colons
    colon_list_pattern = r'(:)\s+(\d+\.\s+)([A-Z])'
    text = re.sub(colon_list_pattern, r'\1\n\n\2\3', text)
    
    # Handle cases where the list starts right after a word without punctuation
    word_list_pattern = r'([a-z])\s+(\d+\.\s+)([A-Z])'
    text = re.sub(word_list_pattern, r'\1\n\n\2\3', text)
    
    # Fix bullet points that are inline
    bullet_pattern = r'([.!?])\s+(•\s+)([A-Z])'
    text = re.sub(bullet_pattern, r'\1\n\n\2\3', text)
    
    # Fix bullet points after colons
    bullet_colon_pattern = r'(:)\s+(•\s+)([A-Z])'
    text = re.sub(bullet_colon_pattern, r'\1\n\n\2\3', text)
    
    # Ensure consistent spacing in numbered lists
    text = re.sub(r'^(\d+\.)\s+', r'\1 ', text, flags=re.MULTILINE)
    
    # Ensure consistent spacing in bullet lists
    text = re.sub(r'^(•)\s+', r'\1 ', text, flags=re.MULTILINE)
    
    return text


def _fix_concatenated_text_after_urls(text: str) -> str:
    """Fix concatenated text that appears after URLs by adding spaces between likely word boundaries."""
    
    # Handle the most common concatenated text patterns after URLs
    concatenation_fixes = [
        # Exact pattern matches - handle both with and without trailing slash
        (r'(https?://[^\s]+/)Thisisthemainwebsiteandthemostcommonwaytoaccessit\.?', 
         r'\1 This is the main website and the most common way to access it.'),
        (r'(https?://[^\s]+)Thisisthemainwebsiteandthemostcommonwaytoaccessit\.?', 
         r'\1 This is the main website and the most common way to access it.'),
        
        # Handle "andyou'rethere" variations - fix the apostrophe issue
        (r'(https?://[^\s]+/)andyou\'?rethere\.?', 
         r'\1 and you\'re there.'),
        (r'(https?://[^\s]+)andyou\'?rethere\.?', 
         r'\1 and you\'re there.'),
        (r'(https?://[^\s]+/)andyourethere\.?', 
         r'\1 and you\'re there.'),
        (r'(https?://[^\s]+)andyourethere\.?', 
         r'\1 and you\'re there.'),
    ]
    
    # Apply each fix
    for pattern, replacement in concatenation_fixes:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text


def _clean_markdown_text(text: str) -> str:
    """Clean markdown text for better rendering."""
    if not text.strip():
        return text
    
    # Fix header formatting issues
    text = re.sub(r'^(#{1,6})\s*([^#\n]+)', r'\1 \2', text, flags=re.MULTILINE)
    
    # Ensure proper spacing around headers
    text = re.sub(r'([^\n])\n(#{1,6}\s)', r'\1\n\n\2', text)
    text = re.sub(r'(#{1,6}[^\n]+)\n([^\n#])', r'\1\n\n\2', text)
    
    # Enhanced list formatting
    # Fix bullet lists
    text = re.sub(r'^(\s*[-*+]\s)', r'\1', text, flags=re.MULTILINE)
    
    # Fix numbered lists - ensure they're on separate lines
    text = re.sub(r'^(\s*\d+\.\s)', r'\1', text, flags=re.MULTILINE)
    
    # Ensure numbered lists have proper line breaks before them
    text = re.sub(r'([^\n])\n(\d+\.\s)', r'\1\n\n\2', text)
    
    # Fix list items that might be running together
    text = re.sub(r'(\d+\.\s[^\n]+)\s+(\d+\.\s)', r'\1\n\n\2', text)
    
    # Clean up excessive spacing but preserve intentional formatting
    text = re.sub(r'\n{4,}', '\n\n\n', text)  # Max 3 newlines for section breaks
    
    return text.strip()


def check_fastapi_backend() -> bool:
    """Return True if the FastAPI backend health endpoint responds with HTTP 200."""
    try:
        response = requests.get(f"{FASTAPI_URL}/health", timeout=FASTAPI_TIMEOUT)
        return response.status_code == 200
    except requests.RequestException:
        return False


def get_processing_capabilities() -> Optional[Dict[str, Any]]:
    """Fetch available processing capabilities from the backend, if any."""
    try:
        response = requests.get(f"{FASTAPI_URL}/api/processing/capabilities", timeout=FASTAPI_TIMEOUT)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.RequestException:
        return None


def get_available_models() -> List[str]:
    """Return a list of models advertised by the backend."""
    try:
        response = requests.get(f"{FASTAPI_URL}/models", timeout=FASTAPI_TIMEOUT)
        payload = response.json()
        return payload.get("models", []) or ["default"]
    except requests.RequestException:
        return ["default"]


def reset_rag_system() -> Tuple[bool, str]:
    """Reset the entire RAG system by clearing all documents and embeddings."""
    try:
        response = requests.post(f"{FASTAPI_URL}/api/rag/reset", timeout=FASTAPI_TIMEOUT)
        if response.status_code == 200:
            result = response.json()
            return True, result.get("message", "RAG system reset successfully")
        else:
            return False, f"Reset failed with status {response.status_code}: {response.text}"
    except requests.RequestException as exc:
        return False, f"Reset Error: {exc}"


def get_rag_status() -> Optional[Dict[str, Any]]:
    """Get current RAG system status."""
    try:
        response = requests.get(f"{FASTAPI_URL}/api/rag/status", timeout=FASTAPI_TIMEOUT)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.RequestException:
        return None


def check_file_duplicates(uploaded_files: List[Any]) -> Optional[Dict[str, Any]]:
    """Check if uploaded files are duplicates without processing them."""
    try:
        files = [("files", (uf.name, uf.getvalue(), uf.type)) for uf in uploaded_files]
        
        response = requests.post(
            f"{FASTAPI_URL}/api/rag/check-duplicates",
            files=files,
            timeout=FASTAPI_TIMEOUT,
        )
        
        if response.status_code == 200:
            return response.json()
        return None
    except requests.RequestException as exc:
        st.error(f"Duplicate Check Error: {exc}")
        return None


def get_processed_files() -> Optional[Dict[str, Any]]:
    """Get information about all processed files."""
    try:
        response = requests.get(f"{FASTAPI_URL}/api/rag/files", timeout=FASTAPI_TIMEOUT)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.RequestException:
        return None


def remove_processed_file(filename: str) -> Tuple[bool, str]:
    """Remove a specific file from the RAG system."""
    try:
        response = requests.delete(f"{FASTAPI_URL}/api/rag/files/{filename}", timeout=FASTAPI_TIMEOUT)
        if response.status_code == 200:
            result = response.json()
            return True, result.get("message", "File removed successfully")
        else:
            return False, f"Failed to remove file: {response.text}"
    except requests.RequestException as exc:
        return False, f"Remove Error: {exc}"


def _animate_tokens(
    placeholder: Optional[Any],
    base_text: str,
    addition: str,
    delay: float = 0.02,
    show_cursor: bool = True,
) -> str:
    """Animate text token-by-token in a Streamlit placeholder, returning the
    concatenated visible text."""
    tokens = re.findall(r"\s+|[^\s]+", addition)
    visible = base_text
    for tok in tokens:
        visible += tok
        if placeholder is not None:
            placeholder.markdown(_unescape_text(visible) + ("▌" if show_cursor else ""))
            time.sleep(delay)
    return visible


# ---------------------------------------------------------------------------
# Backend calls (non-streaming)
# ---------------------------------------------------------------------------

def call_fastapi_chat(message: str, model: str = "default", temperature: float = 0.7) -> Optional[str]:
    """Call the non-streaming chat endpoint and return the response text, if any."""
    try:
        response = requests.post(
            f"{FASTAPI_URL}/api/chat",
            json={"message": message, "model": model, "temperature": temperature},
            timeout=FASTAPI_TIMEOUT,
        )
        if response.status_code == 200:
            return response.json().get("response")
        st.error(f"FastAPI Error: {response.status_code}")
        return None
    except requests.RequestException as exc:
        st.error(f"FastAPI Connection Error: {exc}")
        return None


def call_fastapi_rag_query(
    query: str,
    messages: List[Dict[str, str]],
    model: str,
    system_prompt: str,
    n_results: int = 3,
    use_multi_agent: bool = False,
    use_hybrid_search: bool = False,
) -> Optional[str]:
    """Call the non-streaming RAG endpoint and return raw text, if any."""
    try:
        response = requests.post(
            f"{FASTAPI_URL}/api/rag/query",
            json={
                "query": query,
                "messages": messages,
                "model": model,
                "system_prompt": system_prompt,
                "n_results": n_results,
                "use_multi_agent": use_multi_agent,
                "use_hybrid_search": use_hybrid_search,
            },
            timeout=FASTAPI_TIMEOUT * 2,
        )
        return response.text if response.status_code == 200 else None
    except requests.RequestException as exc:
        st.error(f"RAG API Error: {exc}")
        return None



# ---------------------------------------------------------------------------
# Streaming UI: segregate reasoning during streaming
# ---------------------------------------------------------------------------

class _StreamingSegregator:
    """
    Incrementally separates <think>...</think> (and variants) from the answer
    during streaming and renders both live in the UI.
    - Shows a collapsible "Model Reasoning Process" expander (unless hidden).
    - Streams the answer without the reasoning content.
    This avoids waiting until the end to format the output.
    """
    _OPEN_TAG_RE = re.compile(r"<\s*(think|thinking|reasoning|thought)\s*>", re.IGNORECASE)
    _CLOSE_TAG_RE = re.compile(r"<\s*/\s*(think|thinking|reasoning|thought)\s*>", re.IGNORECASE)
    _ANY_TAG_RE = re.compile(r"(?is)<\s*/?\s*(think|thinking|reasoning|thought)\s*>")

    def __init__(self, placeholder: Optional[Any]):
        self.placeholder = placeholder
        self.in_think = False
        self.pending = ""
        self.thinking_text = ""
        self.answer_text = ""
        self.reasoning_hidden = st.session_state.get("hide_reasoning", False)
        self.reasoning_expanded_default = st.session_state.get("show_reasoning_expanded", False)

        # Pre-allocate layout areas so reasoning (if it appears) stays above the answer.
        if self.placeholder is not None:
            self.root = self.placeholder.container()
            self._thinking_outer = self.root.empty()   # where the expander will appear
            self._separator = self.root.empty()        # horizontal rule between sections
            self._answer_box = self.root.empty()       # main answer stream
        else:
            self.root = None
            self._thinking_outer = None
            self._separator = None
            self._answer_box = None

        self._expander_created = False
        self._reasoning_box = None

    def _ensure_reasoning_ui(self) -> None:
        if self.reasoning_hidden or self._expander_created or self._thinking_outer is None:
            return
        # Build the expander once, at the top of this message block.
        with self._thinking_outer:
            with st.expander("🧠 Model Reasoning Process", expanded=self.reasoning_expanded_default):
                st.markdown("*This section shows the model's internal reasoning and thought process:*")
                st.markdown("---")
                self._reasoning_box = st.empty()
        # Add a separator between reasoning and the streamed answer
        if self._separator is not None:
            self._separator.markdown("---")
        self._expander_created = True

    def _render_reasoning(self) -> None:
        if self.reasoning_hidden or self._reasoning_box is None:
            return
        cleaned = _clean_text_formatting(self.thinking_text)
        cleaned = _clean_markdown_text(cleaned)
        cleaned = sanitize_links(_unescape_text(cleaned))
        self._reasoning_box.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, rgba(63, 81, 181, 0.1), rgba(33, 150, 243, 0.05));
                padding: 1.2rem;
                border-radius: 0.75rem;
                border-left: 4px solid #3f51b5;
                margin: 0.5rem 0;
                font-family: 'SF Pro Text', -apple-system, BlinkMacSystemFont, sans-serif;
            ">
                {cleaned}
            </div>
            """,
            unsafe_allow_html=True
        )

    def _render_answer(self) -> None:
        if self._answer_box is None:
            return
        cleaned = _clean_text_formatting(self.answer_text)
        cleaned = _clean_markdown_text(cleaned)
        cleaned = sanitize_links(_unescape_text(cleaned))
        self._answer_box.markdown(cleaned)

    def _append_text(self, text: str) -> None:
        if not text:
            return
        if self.in_think:
            # Lazily create the reasoning UI when we first see thinking content
            self._ensure_reasoning_ui()
            self.thinking_text += text
            self._render_reasoning()
        else:
            self.answer_text += text
            self._render_answer()

    def feed(self, chunk: str) -> None:
        """
        Feed a new chunk of text from the stream. This method incrementally
        parses <think>...</think> tags and updates the UI accordingly.
        """
        self.pending += chunk

        # Process complete tags found in the buffer
        while True:
            m = self._ANY_TAG_RE.search(self.pending)
            if not m:
                break

            before = self.pending[:m.start()]
            self._append_text(before)

            tag_text = m.group(0)
            is_open = bool(self._OPEN_TAG_RE.fullmatch(tag_text))
            is_close = bool(self._CLOSE_TAG_RE.fullmatch(tag_text))

            if is_open:
                self.in_think = True
                # Ensure UI is ready for reasoning section as soon as we enter it
                self._ensure_reasoning_ui()
            elif is_close:
                self.in_think = False

            # Consume the tag
            self.pending = self.pending[m.end():]

        # Handle any trailing partial tag: keep from last '<' if no closing '>' yet
        last_lt = self.pending.rfind("<")
        if last_lt != -1 and ">" not in self.pending[last_lt:]:
            # Emit everything before the potential tag start
            self._append_text(self.pending[:last_lt])
            self.pending = self.pending[last_lt:]
        else:
            # No partial tag at end; emit all
            self._append_text(self.pending)
            self.pending = ""

# ---------------------------------------------------------------------------
# Backend calls (streaming)
# ---------------------------------------------------------------------------

def stream_fastapi_chat(
    message: str,
    model: str = "default",
    temperature: float = 0.7,
    placeholder: Optional[Any] = None,
) -> Optional[str]:
    """Stream a chat completion from the FastAPI backend into the UI."""
    payload = {"message": message, "model": model, "temperature": temperature}

    try:
        with requests.post(
            f"{FASTAPI_URL}/api/chat/stream",
            json=payload,
            stream=True,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        ) as r:
            r.raise_for_status()

            buf = ""
            segregator = _StreamingSegregator(placeholder) if placeholder is not None else None

            for line in r.iter_lines(decode_unicode=True, chunk_size=1):
                if not line:
                    continue
                if line.startswith("data:"):
                    payload = line[5:].lstrip()
                    if payload.strip() in ("[DONE]", ""):
                        continue
                    line = payload

                chunk = line + "\n"
                buf += chunk
                if segregator is not None:
                    segregator.feed(chunk)

            return buf

    except requests.Timeout:
        st.error("Chat API timed out while streaming.")
    except requests.RequestException as exc:
        st.error(f"Chat API Error: {exc}")

    return None


def stream_local_ollama_chat(
    message: str,
    model: str,
    temperature: float,
    placeholder: Optional[Any] = None,
) -> str:
    """Stream a chat completion from a local Ollama service into the UI."""
    async def _inner() -> str:
        acc = ""
        segregator = _StreamingSegregator(placeholder) if placeholder is not None else None
        async for chunk in stream_ollama(
            [{"role": "user", "content": message}], model=model, temperature=temperature
        ):
            acc += chunk
            if segregator is not None:
                segregator.feed(chunk)
        return acc

    return asyncio.run(_inner())


def stream_fastapi_rag_query(
    query: str,
    messages: List[Dict[str, str]],
    model: str,
    system_prompt: str,
    n_results: int = 3,
    use_multi_agent: bool = False,
    use_hybrid_search: bool = False,
    placeholder: Optional[Any] = None,
) -> Optional[str]:
    """Stream a RAG response from the FastAPI backend into the UI."""
    payload = {
        "query": query,
        "messages": messages,
        "model": model,
        "system_prompt": system_prompt,
        "n_results": n_results,
        "use_multi_agent": use_multi_agent,
        "use_hybrid_search": use_hybrid_search,
    }

    try:
        with requests.post(
            f"{FASTAPI_URL}/api/rag/query",
            json=payload,
            stream=True,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        ) as r:
            r.raise_for_status()

            buf = ""
            segregator = _StreamingSegregator(placeholder) if placeholder is not None else None

            for line in r.iter_lines(decode_unicode=True, chunk_size=1):
                if not line:
                    continue
                if line.startswith("data:"):
                    payload = line[5:].lstrip()
                    if payload.strip() in ("[DONE]", ""):
                        continue
                    line = payload

                chunk = line + "\n"
                buf += chunk
                if segregator is not None:
                    segregator.feed(chunk)
            return buf

    except requests.Timeout:
        st.error("RAG API timed out while streaming. Try narrowing the query or lowering `n_results`.")
    except requests.RequestException as exc:
        st.error(f"RAG API Error: {exc}")

    return None


# ---------------------------------------------------------------------------
# RAG configuration / uploads
# ---------------------------------------------------------------------------

def save_rag_configuration(provider: str, embedding_model: str, chunk_size: int) -> Tuple[bool, str]:
    """Persist RAG configuration to the backend. Returns (success, message)."""
    try:
        response = requests.post(
            f"{FASTAPI_URL}/api/rag/save_config",
            json={"provider": provider, "embedding_model": embedding_model, "chunk_size": chunk_size},
            timeout=FASTAPI_TIMEOUT,
        )
        return (response.status_code == 200, "OK" if response.status_code == 200 else response.text)
    except requests.RequestException as exc:
        return False, f"Configuration Error: {exc}"


def upload_files_to_fastapi_enhanced(uploaded_files: List[Any], chunk_size: int = 500) -> Tuple[bool, Dict[str, Any]]:
    """Upload files to the backend for enhanced processing. Returns (success, result)."""
    if not uploaded_files:
        return False, {}

    try:
        files = [("files", (uf.name, uf.getvalue(), uf.type)) for uf in uploaded_files]

        progress = st.empty()
        progress.info("📤 Uploading files to backend...")

        response = requests.post(
            f"{FASTAPI_URL}/api/rag/upload",
            files=files,
            data={"chunk_size": chunk_size},
            timeout=300,
        )

        progress.empty()

        if response.status_code == 200:
            return True, response.json()

        st.error(f"Upload failed with status {response.status_code}: {response.text}")
        return False, {}

    except requests.RequestException as exc:
        st.error(f"Upload Error: {exc}")
        return False, {}


def display_upload_results(results: Dict[str, Any]) -> None:
    """Render a detailed summary of enhanced processing results with duplicate detection."""
    if not results:
        return

    summary = results.get("summary", {})
    file_results = results.get("results", [])

    # Display summary statistics
    if summary:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Files", summary.get("total_files", 0))
        with col2:
            st.metric("Successful", summary.get("successful", 0))
        with col3:
            st.metric("Duplicates", summary.get("duplicates", 0))
        with col4:
            st.metric("Total Chunks", summary.get("total_chunks_created", 0))

    if file_results:
        st.markdown("**📋 Detailed Results:**")

        successful = [r for r in file_results if r.get("status") == "success"]
        duplicates = [r for r in file_results if r.get("status") == "duplicate"]
        failed = [r for r in file_results if r.get("status") == "error"]

        if successful:
            with st.expander(f"✅ Successfully Processed ({len(successful)} files)", expanded=True):
                for r in successful:
                    processing_type = r.get("processing_type", "standard")
                    chunks = r.get("chunks_created", 0)

                    if processing_type == "enhanced_csv":
                        st.success(f"📊 **{r['filename']}** — Enhanced CSV Analysis ({chunks} intelligent chunks)")
                        details = r.get("details", {}) or {}
                        st.markdown(f"   • Column analysis: {'✅' if details.get('column_analysis') else '❌'}")
                        st.markdown(f"   • Statistical summaries: {'✅' if details.get('statistical_summaries') else '❌'}")
                        st.markdown(f"   • Intelligent chunking: {'✅' if details.get('intelligent_chunking') else '❌'}")

                    elif processing_type == "enhanced_image":
                        st.success(f"🖼️ **{r['filename']}** — Enhanced Image Processing ({chunks} content chunks)")
                        details = r.get("details", {}) or {}
                        st.markdown(f"   • OCR text extraction: {'✅' if details.get('ocr_extraction') else '❌'}")
                        st.markdown(f"   • Content analysis: {'✅' if details.get('content_analysis') else '❌'}")
                        st.markdown(f"   • Structure detection: {'✅' if details.get('structured_detection') else '❌'}")

                    else:
                        st.success(f"📄 **{r['filename']}** — Standard Processing ({chunks} chunks)")
                    if r.get("message"):
                        st.markdown(f"   *{r['message']}*")

        if duplicates:
            with st.expander(f"⚠️ Duplicate Files Skipped ({len(duplicates)} files)", expanded=True):
                for r in duplicates:
                    st.warning(f"**{r['filename']}**: {r.get('message', 'Duplicate detected')}")

        if failed:
            with st.expander(f"❌ Processing Errors ({len(failed)} files)", expanded=True):
                for r in failed:
                    st.error(f"**{r['filename']}**: {r.get('message', 'Unknown error')}")


def get_file_type_info() -> Dict[str, Any]:
    """Return supported file extensions for the enhanced uploader."""
    return {
        "supported_extensions": [
            # Documents
            "pdf", "doc", "docx", "ppt", "pptx", "rtf", "odt", "epub",
            # Spreadsheets
            "xls", "xlsx", "csv", "tsv",
            # Text files
            "txt", "md", "html", "htm", "org", "rst",
            # Images
            "png", "jpg", "jpeg", "bmp", "tiff", "webp", "gif", "heic", "svg",
        ]
    }


# ---------------------------------------------------------------------------
# App definition
# ---------------------------------------------------------------------------

st.set_page_config(page_title="LLM WebUI", layout="wide", initial_sidebar_state="expanded")

if "backend_available" not in st.session_state:
    st.session_state.backend_available = check_fastapi_backend()

if "capabilities" not in st.session_state:
    st.session_state.capabilities = get_processing_capabilities() if st.session_state.backend_available else None

backend_available: bool = st.session_state.backend_available
capabilities: Optional[Dict[str, Any]] = st.session_state.capabilities
backend_mode = "FastAPI" if backend_available else "Local"

st.sidebar.title("🤖 LLM WebUI")
st.sidebar.markdown("---")

page = st.sidebar.selectbox("Navigate to:", ["💬 Chat", "📚 RAG"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("🔧 System Status")
if backend_available:
    st.sidebar.success("✅ FastAPI Backend Connected")
else:
    st.sidebar.warning("⚠️ Using Local Backend")

# ---------------------------- Chat Page ------------------------------------

if page == "💬 Chat":
    st.title("💬 LLM Chat Interface")
    st.sidebar.markdown("---")
    st.sidebar.subheader("💬 Chat Configuration")

    st.session_state.setdefault("chat_messages", [])

    available_models = get_available_models() if backend_available else ["gemma3n:e2b"]

    selected_model = st.sidebar.selectbox("Select Model:", available_models)
    temperature = st.sidebar.slider("Temperature:", 0.0, 1.0, 0.7, 0.1)
    
    # Reasoning display settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 Reasoning Display")
    st.session_state.show_reasoning_expanded = st.sidebar.checkbox(
        "Expand reasoning by default",
        value=st.session_state.get("show_reasoning_expanded", False),
        help="Show model reasoning sections expanded by default"
    )
    
    hide_reasoning = st.sidebar.checkbox(
        "Hide reasoning completely",
        value=st.session_state.get("hide_reasoning", False),
        help="Completely hide reasoning sections from responses"
    )
    st.session_state.hide_reasoning = hide_reasoning

    if st.sidebar.button("🗑️ Clear Chat History", type="secondary"):
        st.session_state.chat_messages.clear()
        st.rerun()

    if st.session_state.chat_messages:
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                render_llm_text(msg["content"])
    else:
        st.markdown(
            """
            <div style='text-align:center; color:gray; font-size:1.2em; padding-top:16rem; padding-bottom:4rem;'>
                How can I help you today?
            </div>
            """,
            unsafe_allow_html=True,
        )

    user_input = st.chat_input("Ask Anything")

    if user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            render_llm_text(user_input)

        with st.chat_message("assistant"):
            out = st.empty()
            with st.spinner("Thinking..."):
                if backend_available:
                    reply = stream_fastapi_chat(
                        message=user_input,
                        model=selected_model,
                        temperature=temperature,
                        placeholder=out,
                    )
                else:
                    reply = stream_local_ollama_chat(
                        message=user_input,
                        model=selected_model,
                        temperature=temperature,
                        placeholder=out,
                    )

        if reply:
            # UI already rendered incrementally by the streaming function.
            st.session_state.chat_messages.append({"role": "assistant", "content": reply})
        else:
            st.error("❌ No response received.")
            st.session_state.chat_messages.append({"role": "assistant", "content": "No response."})

        st.rerun()

# ----------------------------- RAG Page ------------------------------------

elif page == "📚 RAG":
    st.title("📚 RAG Interface")

    if backend_available:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📚 RAG Configuration")

        st.session_state.setdefault("config_saved", False)
        st.session_state.setdefault("config_message", "")
        st.session_state.setdefault("rag_messages", [])

        provider = st.sidebar.selectbox(
            "Embedding Provider:",
            ["Ollama", "HuggingFace"],
            index=0,
            help="Provider used for document embeddings",
        )

        if provider == "Ollama":
            available_embedding_models = [m for m in get_available_models() if "embed" in m.lower()] or get_available_models()
            embedding_model = st.sidebar.selectbox(
                "Ollama Embedding Model:",
                available_embedding_models,
                help="Select an embedding model from your Ollama installation",
            )
        else:
            embedding_model = st.sidebar.text_input(
                "HuggingFace Model Repo:",
                value="sentence-transformers/all-MiniLM-L6-v2",
                help="Enter a HuggingFace model repository path",
            )

        chunk_size = st.sidebar.number_input(
            "Chunk Size:",
            value=500,
            min_value=100,
            max_value=2000,
            step=100,
            help="Text chunk size for standard processing",
        )

        n_results = st.sidebar.number_input(
            "Number of Results:",
            value=3,
            min_value=1,
            max_value=10,
            help="Number of relevant chunks to retrieve",
        )

        use_multi_agent = st.sidebar.checkbox(
            "Use Multi-Agent Orchestration (CrewAI)",
            value=False,
            help="Enable multi-agent RAG processing",
        )

        if st.sidebar.button("💾 Save Configuration", type="primary"):
            with st.spinner("Saving configuration..."):
                success, message = save_rag_configuration(provider, embedding_model, chunk_size)
                st.session_state.config_saved = success
                st.session_state.config_message = "✅ Configuration saved successfully!" if success else f"❌ Failed to save configuration: {message}"

        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 Search Configuration")

        use_hybrid_search = st.sidebar.checkbox(
            "🔍 Enable Hybrid Search (BM25 + Vector)",
            value=False,
            help="Combine keyword BM25 with semantic vector search",
            disabled=not st.session_state.config_saved,
        )

        if st.session_state.config_message:
            (st.success if st.session_state.config_saved else st.error)(st.session_state.config_message)

        # RAG System Status and Reset
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔄 System Management")
        
        # Display current status
        rag_status = get_rag_status()
        if rag_status:
            if rag_status["status"] == "ready":
                st.sidebar.success(f"✅ System Ready")
                st.sidebar.write(f"📁 Files: {rag_status.get('uploaded_files', 0)}")
                st.sidebar.write(f"📊 Chunks: {rag_status.get('indexed_chunks', 0)}")
            else:
                st.sidebar.warning(f"⚠️ Status: {rag_status['status']}")
        else:
            st.sidebar.error("❌ Cannot connect to backend")

        # Reset button with confirmation
        st.session_state.setdefault("show_reset_confirm", False)
        
        if st.sidebar.button("🗑️ Reset System", 
                           type="secondary", 
                           help="Clear all uploaded documents, embeddings, and chat history",
                           disabled=not st.session_state.config_saved):
            st.session_state.show_reset_confirm = True

        # Reset confirmation dialog
        if st.session_state.show_reset_confirm:
            st.sidebar.markdown("---")
            st.sidebar.warning("⚠️ **Confirm Reset**")
            st.sidebar.write("This will permanently delete:")
            st.sidebar.write("• All uploaded documents")
            st.sidebar.write("• All embeddings and indexes")
            st.sidebar.write("• Current chat history")
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button("✅ Confirm", type="primary", key="confirm_reset"):
                    with st.spinner("Resetting RAG system..."):
                        success, message = reset_rag_system()
                        if success:
                            st.success(f"✅ {message}")
                            # Clear chat history
                            st.session_state.rag_messages.clear()
                            st.session_state.show_reset_confirm = False
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
                            st.session_state.show_reset_confirm = False
            
            with col2:
                if st.button("❌ Cancel", key="cancel_reset"):
                    st.session_state.show_reset_confirm = False
                    st.rerun()

        selected_model_rag = st.selectbox(
            "Select Base Model:",
            get_available_models(),
            help="Language model used for generating responses",
        )
        
        # Reasoning display settings for RAG
        st.sidebar.markdown("---")
        st.sidebar.subheader("🧠 Reasoning Display")
        st.session_state.show_reasoning_expanded = st.sidebar.checkbox(
            "Expand reasoning by default",
            value=st.session_state.get("show_reasoning_expanded", False),
            help="Show model reasoning sections expanded by default",
            key="rag_reasoning_expanded"
        )
        
        hide_reasoning = st.sidebar.checkbox(
            "Hide reasoning completely",
            value=st.session_state.get("hide_reasoning", False),
            help="Completely hide reasoning sections from responses",
            key="rag_hide_reasoning"
        )
        st.session_state.hide_reasoning = hide_reasoning

        st.subheader("📁 Upload Documents")

        # Show current knowledge base status
        rag_status = get_rag_status()
        if rag_status and rag_status["status"] == "ready":
            if rag_status.get("uploaded_files", 0) > 0:
                st.info(f"📊 Current Knowledge Base: {rag_status['uploaded_files']} files, {rag_status['indexed_chunks']} chunks indexed")
            else:
                st.info("📭 Knowledge base is empty. Upload documents to get started.")

        if not st.session_state.config_saved:
            st.warning("⚠️ Please save your configuration before uploading documents.")

        file_info = get_file_type_info()

        uploaded_files = st.file_uploader(
            "Upload Documents for Enhanced Processing",
            type=file_info["supported_extensions"],
            accept_multiple_files=True,
            help="Optionally upload CSVs, images, or other documents for enhanced processing",
            disabled=not st.session_state.config_saved,
        )

        # Add duplicate check button
        if uploaded_files:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔍 Check for Duplicates", disabled=not st.session_state.config_saved):
                    with st.spinner("Checking for duplicate files..."):
                        duplicate_results = check_file_duplicates(uploaded_files)
                        if duplicate_results:
                            st.subheader("📋 Duplicate Check Results")
                            
                            duplicates = [r for r in duplicate_results["results"] if r["is_duplicate"]]
                            new_files = [r for r in duplicate_results["results"] if not r["is_duplicate"]]
                            
                            if duplicates:
                                st.warning(f"⚠️ Found {len(duplicates)} duplicate files:")
                                for dup in duplicates:
                                    st.write(f"• **{dup['filename']}**: {dup['reason']}")
                            
                            if new_files:
                                st.success(f"✅ {len(new_files)} new files ready for processing:")
                                for new_file in new_files:
                                    st.write(f"• **{new_file['filename']}**: {new_file['reason']}")
                        else:
                            st.error("❌ Failed to check for duplicates")
            
            with col2:
                if st.button("📤 Upload & Process", type="primary", disabled=not st.session_state.config_saved):
                    with st.spinner(f"Processing {len(uploaded_files)} files..."):
                        success, results = upload_files_to_fastapi_enhanced(uploaded_files, chunk_size)
                        if success and results:
                            st.success("✅ Enhanced processing completed!")
                            display_upload_results(results)
                            st.session_state.rag_messages.clear()
                            st.info("💡 Chat history cleared to reflect new knowledge base")
                            logger.info("Enhanced file processing completed successfully")
                            # Refresh the page to show updated status
                            st.rerun()
                        else:
                            st.error("❌ Enhanced processing failed. See details above.")

        # File Management Section
        st.subheader("📂 File Management")
        
        processed_files_info = get_processed_files()
        if processed_files_info and processed_files_info.get("total_files", 0) > 0:
            st.write(f"**Total Files**: {processed_files_info['total_files']} | **Total Chunks**: {processed_files_info['total_chunks']}")
            
            with st.expander("📋 View Processed Files", expanded=False):
                for file_info in processed_files_info.get("files", []):
                    col1, col2, col3 = st.columns([3, 2, 1])
                    
                    with col1:
                        st.write(f"**{file_info['filename']}**")
                        st.caption(f"Processed: {file_info['processed_at']}")
                    
                    with col2:
                        st.write(f"📊 {file_info['chunk_count']} chunks")
                    
                    with col3:
                        if st.button("🗑️", key=f"remove_{file_info['filename']}", help=f"Remove {file_info['filename']}"):
                            with st.spinner(f"Removing {file_info['filename']}..."):
                                success, message = remove_processed_file(file_info['filename'])
                                if success:
                                    st.success(f"✅ {message}")
                                    st.rerun()
                                else:
                                    st.error(f"❌ {message}")
        else:
            st.info("📭 No files have been processed yet.")

        st.subheader("⚙️ System Configuration")
        system_prompt = st.text_area(
            "System Prompt:",
            value=(
                "You are a helpful assistant with access to enhanced multimodal document knowledge. "
                "Use detailed analysis from CSV files, extracted text from images, and comprehensive document "
                "content to provide accurate answers. If the answer requires information not in the context, "
                "clearly state that."
            ),
            height=120,
            help="System prompt optimized for enhanced multimodal processing",
        )

        st.subheader("🔍 Query Your Enhanced Knowledge Base")

        for message in st.session_state.rag_messages:
            with st.chat_message(message["role"]):
                render_llm_text(message["content"])

        rag_input = st.chat_input("Ask about your documents, data analysis, image content, or any uploaded materials...")

        if rag_input:
            st.session_state.rag_messages.append({"role": "user", "content": rag_input})

            with st.chat_message("user"):
                render_llm_text(rag_input)

            history = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.rag_messages[:-1]]

            with st.chat_message("assistant"):
                status = "Searching enhanced knowledge base"
                if use_hybrid_search:
                    status += " (hybrid search enabled)"
                if use_multi_agent:
                    status += " with multi-agent orchestration"
                status += "..."

                out = st.empty()
                with st.spinner(status):
                    response_text = stream_fastapi_rag_query(
                        query=rag_input,
                        messages=history,
                        model=selected_model_rag,
                        system_prompt=system_prompt,
                        n_results=n_results,
                        use_multi_agent=use_multi_agent,
                        use_hybrid_search=use_hybrid_search,
                        placeholder=out,
                    )

                if response_text:
                    # UI already rendered incrementally by the streaming function.
                    st.session_state.rag_messages.append({"role": "assistant", "content": response_text})
                else:
                    err = "❌ No response received. Please check your query and try again."
                    st.error(err)
                    st.session_state.rag_messages.append({"role": "assistant", "content": err})

    else:
        st.info("📡 Enhanced RAG functionality requires the FastAPI backend. Please start the backend server.")

# ------------------------------ Footer -------------------------------------

st.sidebar.markdown("---")
if backend_available:
    if capabilities and capabilities.get("status") == "ready":
        st.sidebar.write("🟢 Enhanced processing ready")
    else:
        st.sidebar.write("🟡 Basic functionality available")
else:
    st.sidebar.write("🔴 Limited functionality (backend offline)")