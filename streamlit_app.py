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
    - Remove all internal spaces.
    - Prepend https:// for leading www.
    """
    u = url.strip()
    # Normalize spaced scheme separators, keep lowercase scheme
    u = re.sub(r"(?i)https?\s*:\s*/\s*/", lambda m: m.group(0).lower().replace(" ", ""), u)
    # Remove any remaining spaces inside the URL
    u = re.sub(r"\s+", "", u)
    if u.startswith("www."):
        u = "https://" + u
    return u


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

    # 2) Normalize bare URLs like: https :// gemma. google. com /path
    #    Match from scheme through allowed URL chars, permitting literal spaces we will remove.
    url_char_class = r"-A-Za-z0-9\._~:/\?#\[\]@!\$&'\(\)\*\+,;=%"
    bare_url_pattern = rf"(?i)https?\s*:\s*/\s*/(?:[{url_char_class}]| )+"

    def _bare_url_repl(m: re.Match) -> str:
        return _clean_url(m.group(0))

    text = re.sub(bare_url_pattern, _bare_url_repl, text)
    return text


def render_llm_text(s: str) -> None:
    """Render mixed Markdown and fenced code blocks in Streamlit."""
    text = _unescape_text(s).strip()
    code_fence = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)

    pos = 0
    last_end = 0
    rendered_any = False

    for match in code_fence.finditer(text):
        before = text[pos:match.start()]
        if before.strip():
            st.markdown(sanitize_links(before))
            rendered_any = True

        lang = (match.group(1) or "").strip()
        code = match.group(2).strip("\n")
        st.code(code, language=lang if lang else None)
        rendered_any = True

        pos = match.end()
        last_end = match.end()

    tail = text[last_end:] if last_end else text
    if tail.strip() or not rendered_any:
        st.markdown(sanitize_links(tail))


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
            for line in r.iter_lines(decode_unicode=True, chunk_size=1):
                if not line:
                    continue
                if line.startswith("data:"):
                    line = line[5:].lstrip()

                buf = _animate_tokens(placeholder, buf, line + "\n") if placeholder else (buf + line + "\n")
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
        async for chunk in stream_ollama(
            [{"role": "user", "content": message}], model=model, temperature=temperature
        ):
            acc += chunk
            if placeholder is not None:
                placeholder.markdown(_unescape_text(acc))
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
            for line in r.iter_lines(decode_unicode=True, chunk_size=1):
                if not line:
                    continue
                if line.startswith("data:"):
                    line = line[5:].lstrip()

                buf = _animate_tokens(placeholder, buf, line + "\n") if placeholder else (buf + line + "\n")
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
    """Render a detailed summary of enhanced processing results."""
    if not results:
        return

    summary = results.get("summary", {})
    file_results = results.get("results", [])

    if summary.get("processing_types"):
        st.markdown("**🔧 Processing Types Used:**")
        badges: List[str] = []
        for ptype in summary["processing_types"]:
            if ptype == "enhanced_csv":
                badges.append("📊 Enhanced CSV Analysis")
            elif ptype == "enhanced_image":
                badges.append("🖼️ Enhanced Image Processing")
            elif ptype == "standard":
                badges.append("📄 Standard Processing")
            else:
                badges.append(f"⚙️ {ptype}")
        st.markdown(" • ".join(badges))

    if file_results:
        st.markdown("**📋 Detailed Results:**")

        successful = [r for r in file_results if r.get("status") == "success"]
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
                        st.success(f"📄 **{r['filename']}** — Standard Processing")
                    if r.get("message"):
                        st.markdown(f"   *{r['message']}*")

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
            out.markdown(sanitize_links(_unescape_text(reply)))
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

        selected_model_rag = st.selectbox(
            "Select Base Model:",
            get_available_models(),
            help="Language model used for generating responses",
        )

        st.subheader("📁 Upload Documents")

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

        if uploaded_files and st.button("📤 Upload & Process with Enhanced Extraction", type="primary", disabled=not st.session_state.config_saved):
            with st.spinner(f"Processing {len(uploaded_files)} files..."):
                success, results = upload_files_to_fastapi_enhanced(uploaded_files, chunk_size)
                if success and results:
                    st.success("✅ Enhanced processing completed!")
                    display_upload_results(results)
                    st.session_state.rag_messages.clear()
                    st.info("💡 Chat history cleared to reflect new knowledge base")
                    logger.info("Enhanced file processing completed successfully")
                else:
                    st.error("❌ Enhanced processing failed. See details above.")

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
                st.write(message["content"])

        rag_input = st.chat_input("Ask about your documents, data analysis, image content, or any uploaded materials...")

        if rag_input:
            st.session_state.rag_messages.append({"role": "user", "content": rag_input})

            with st.chat_message("user"):
                st.write(rag_input)

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
                    out.markdown(sanitize_links(_unescape_text(response_text)))
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