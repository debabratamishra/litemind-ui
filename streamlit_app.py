"""
Enhanced Streamlit front-end for the LLM WebUI with comprehensive multimodal capabilities
Includes enhanced CSV and image processing with real-time feedback
"""

from __future__ import annotations
import asyncio
import requests
from pathlib import Path
import nest_asyncio
import streamlit as st

# === Rendering helpers ===
import re, json

def _unescape_text(s: str) -> str:
    """Turn visible escape sequences (like '\\n', '\\t', '\\u2713') into real characters
    without breaking already-correct text."""
    if not isinstance(s, str):
        s = str(s)
    # Strip accidental surrounding quotes
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    # Try json trick to unescape safely; fall back to targeted replacements
    try:
        s = json.loads(f'"{s}"')
    except Exception:
        s = s.replace('\\\\n', '\n').replace('\\\\t', '\t')
        s = s.replace('\\n', '\n').replace('\\t', '\t')
    # Normalize Windows newlines just in case
    s = s.replace('\r\n', '\n')
    return s

def render_llm_text(s: str):
    """Render mixed Markdown + fenced code blocks nicely in Streamlit."""
    s = _unescape_text(s).strip()

    # If there are fenced code blocks, render alternating markdown/code.
    code_fence = re.compile(r"```(\w+)?\\n(.*?)```", re.DOTALL)
    pos = 0
    last_match_end = 0
    rendered_any = False
    for m in code_fence.finditer(s):
        before = s[pos:m.start()]
        if before.strip():
            st.markdown(before)
            rendered_any = True
        lang = (m.group(1) or "").strip()
        code = m.group(2).strip("\n")
        st.code(code, language=lang if lang else None)
        rendered_any = True
        pos = m.end()
        last_match_end = m.end()
    # Remainder
    tail = s[last_match_end:] if last_match_end else s
    if tail.strip() or not rendered_any:
        # Use markdown so headings, lists, tables render properly
        st.markdown(tail)

from app.services.ollama import stream_ollama
from streamlit.components.v1 import html
import time
import logging
import io
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure FastAPI backend
FASTAPI_URL = "http://localhost:8000"
FASTAPI_TIMEOUT = 60  # seconds

def check_fastapi_backend():
    """Check if FastAPI backend is available"""
    try:
        response = requests.get(f"{FASTAPI_URL}/health", timeout=FASTAPI_TIMEOUT)
        return response.status_code == 200
    except (requests.exceptions.RequestException, requests.exceptions.Timeout):
        return False

def get_processing_capabilities():
    """Get enhanced processing capabilities from backend"""
    try:
        response = requests.get(f"{FASTAPI_URL}/api/processing/capabilities", timeout=FASTAPI_TIMEOUT)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def call_fastapi_chat(message: str, model: str = "default", temperature: float = 0.7):
    """Call FastAPI chat endpoint"""
    try:
        response = requests.post(
            f"{FASTAPI_URL}/api/chat",
            json={"message": message, "model": model, "temperature": temperature},
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            st.error(f"FastAPI Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"FastAPI Connection Error: {str(e)}")
        return None


def call_fastapi_rag_query(query: str, messages: list, model: str, system_prompt: str, n_results: int = 3, use_multi_agent: bool = False, use_hybrid_search: bool = False):
    """Call FastAPI RAG query endpoint with hybrid search support"""
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
                "use_hybrid_search": use_hybrid_search
            },
            timeout=120
        )
        return response.text if response.status_code == 200 else None
    except Exception as e:
        st.error(f"RAG API Error: {str(e)}")
        return None


# Streaming helper for RAG queries
def stream_fastapi_rag_query(
    query: str,
    messages: list,
    model: str,
    system_prompt: str,
    n_results: int = 3,
    use_multi_agent: bool = False,
    use_hybrid_search: bool = False,
    placeholder=None,
):
    """Stream RAG response from FastAPI and update the UI incrementally."""
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
        # Longer read timeout so long answers don't error out
        with requests.post(
            f"{FASTAPI_URL}/api/rag/query",
            json=payload,
            stream=True,
            timeout=(5, 600),  # (connect, read)
        ) as r:
            r.raise_for_status()

            buf = ""
            # Stream line-by-line; works for chunked text/plain and SSE-like lines
            for line in r.iter_lines(decode_unicode=True):
                if line is None:
                    continue
                if not line:
                    # keep-alive / empty line
                    continue

                # Optional: strip SSE "data:" prefix if backend uses it
                if line.startswith("data:"):
                    line = line[5:].lstrip()

                buf += line + "\n"

                # Update UI incrementally
                if placeholder is not None:
                    placeholder.markdown(_unescape_text(buf))

            return buf

    except requests.exceptions.Timeout:
        st.error("RAG API timed out while streaming. Try narrowing the query or lowering `n_results`.")
    except requests.exceptions.RequestException as e:
        st.error(f"RAG API Error: {e}")

    return None

async def call_local_ollama(messages, model: str = "default", temperature: float = 0.7):
    """Fallback to local Ollama service"""
    response = ""
    async for chunk in stream_ollama(messages, model=model, temperature=temperature):
        response += chunk
    return response

def save_rag_configuration(provider: str, embedding_model: str, chunk_size: int):
    """Save RAG configuration to FastAPI backend"""
    try:
        response = requests.post(
            f"{FASTAPI_URL}/api/rag/save_config",
            json={
                "provider": provider,
                "embedding_model": embedding_model,
                "chunk_size": chunk_size
            },
            timeout=60
        )
        
        return response.status_code == 200, response.text
    except Exception as e:
        return False, f"Configuration Error: {str(e)}"

def upload_files_to_fastapi_enhanced(uploaded_files, chunk_size: int = 500):
    """Enhanced upload function with detailed progress and results"""
    if not uploaded_files:
        return False, {}
    
    try:
        files = [("files", (file.name, file.getvalue(), file.type)) for file in uploaded_files]
        
        # Create progress container
        progress_container = st.empty()
        progress_container.info("📤 Uploading files to backend...")
        
        response = requests.post(
            f"{FASTAPI_URL}/api/rag/upload",
            files=files,
            data={"chunk_size": chunk_size},
            timeout=300  # 5 minutes for large files
        )
        
        if response.status_code == 200:
            results = response.json()
            progress_container.empty()
            return True, results
        else:
            progress_container.empty()
            st.error(f"Upload failed with status {response.status_code}: {response.text}")
            return False, {}
    except Exception as e:
        st.error(f"Upload Error: {str(e)}")
        return False, {}

def display_upload_results(results: dict):
    """Display detailed upload results with enhanced processing info"""
    if not results:
        return
    
    summary = results.get("summary", {})
    file_results = results.get("results", [])
    
    # Processing types used
    if summary.get("processing_types"):
        st.markdown("**🔧 Processing Types Used:**")
        processing_types = summary["processing_types"]
        type_badges = []
        for ptype in processing_types:
            if ptype == "enhanced_csv":
                type_badges.append("📊 Enhanced CSV Analysis")
            elif ptype == "enhanced_image":
                type_badges.append("🖼️ Enhanced Image Processing")
            elif ptype == "standard":
                type_badges.append("📄 Standard Processing")
            else:
                type_badges.append(f"⚙️ {ptype}")
        st.markdown(" • ".join(type_badges))
    
    # Detailed results
    if file_results:
        st.markdown("**📋 Detailed Results:**")
        
        successful_files = [r for r in file_results if r["status"] == "success"]
        failed_files = [r for r in file_results if r["status"] == "error"]
        
        if successful_files:
            with st.expander(f"✅ Successfully Processed ({len(successful_files)} files)", expanded=True):
                for result in successful_files:
                    processing_type = result.get("processing_type", "standard")
                    chunks = result.get("chunks_created", 0)
                    
                    # Enhanced display based on processing type
                    if processing_type == "enhanced_csv":
                        st.success(f"📊 **{result['filename']}** - Enhanced CSV Analysis ({chunks} intelligent chunks)")
                        details = result.get("details", {})
                        if details:
                            st.markdown(f"   • Column analysis: {'✅' if details.get('column_analysis') else '❌'}")
                            st.markdown(f"   • Statistical summaries: {'✅' if details.get('statistical_summaries') else '❌'}")
                            st.markdown(f"   • Intelligent chunking: {'✅' if details.get('intelligent_chunking') else '❌'}")
                    
                    elif processing_type == "enhanced_image":
                        st.success(f"🖼️ **{result['filename']}** - Enhanced Image Processing ({chunks} content chunks)")
                        details = result.get("details", {})
                        if details:
                            st.markdown(f"   • OCR text extraction: {'✅' if details.get('ocr_extraction') else '❌'}")
                            st.markdown(f"   • Content analysis: {'✅' if details.get('content_analysis') else '❌'}")
                            st.markdown(f"   • Structure detection: {'✅' if details.get('structured_detection') else '❌'}")
                    
                    else:
                        st.success(f"📄 **{result['filename']}** - Standard Processing")
                    
                    if result.get("message"):
                        st.markdown(f"   *{result['message']}*")
        
        if failed_files:
            with st.expander(f"❌ Processing Errors ({len(failed_files)} files)", expanded=True):
                for result in failed_files:
                    st.error(f"**{result['filename']}**: {result.get('message', 'Unknown error')}")

def get_file_type_info():
    """Return information about supported file types with enhanced capabilities"""
    return {
        "supported_extensions": [
            # Documents
            'pdf', 'doc', 'docx', 'ppt', 'pptx', 'rtf', 'odt', 'epub',
            # Spreadsheets  
            'xls', 'xlsx', 'csv', 'tsv',
            # Text files
            'txt', 'md', 'html', 'htm', 'org', 'rst',
            # Images
            'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp', 'gif', 'heic', 'svg'
        ]
    }

# Initialize Streamlit app
st.set_page_config(
    page_title="LLM WebUI", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check backend status and capabilities
if 'backend_available' not in st.session_state:
    st.session_state.backend_available = check_fastapi_backend()

if 'capabilities' not in st.session_state:
    if st.session_state.backend_available:
        st.session_state.capabilities = get_processing_capabilities()
    else:
        st.session_state.capabilities = None

backend_available = st.session_state.backend_available
backend_mode = "FastAPI" if backend_available else "Local"
capabilities = st.session_state.capabilities

# Page navigation
st.sidebar.title("🤖 LLM WebUI")
st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Navigate to:",
    ["💬 Chat", "📚 RAG"],
    index=0
)

# Backend status in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 System Status")
if backend_available:
    st.sidebar.success("✅ FastAPI Backend Connected")
else:
    st.sidebar.warning("⚠️ Using Local Backend")

# ============ CHAT PAGE ============
if page == "💬 Chat":
    st.title("💬 LLM Chat Interface")

    # Sidebar configuration
    st.sidebar.markdown("---")
    st.sidebar.subheader("💬 Chat Configuration")

    # Initialize session storage
    st.session_state.setdefault("chat_messages", [])

    # Available models
    if backend_available:
        try:
            available_models = requests.get(f"{FASTAPI_URL}/models").json().get("models", ["default"])
        except Exception:
            available_models = ["default"]
    else:
        available_models = ["gemma3n:e2b"]

    selected_model = st.sidebar.selectbox("Select Model:", available_models)
    temperature = st.sidebar.slider("Temperature:", 0.0, 1.0, 0.7, 0.1)

    if st.sidebar.button("🗑️ Clear Chat History", type="secondary"):
        st.session_state.chat_messages.clear()
        st.rerun()

    # Conversation area
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

    # Input bar
    user_input = st.chat_input("Ask Anything")
    if user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            if backend_available:
                reply = call_fastapi_chat(user_input, selected_model, temperature)
            else:
                reply = asyncio.run(
                    call_local_ollama([{"role": "user", "content": user_input}], selected_model, temperature)
                )
        if reply:
            st.session_state.chat_messages.append({"role": "assistant", "content": reply})
        st.rerun()

# ============ RAG PAGE ============
elif page == "📚 RAG":
    st.title("📚 RAG Interface")
    
    if backend_available:
        # RAG-specific sidebar configuration
        st.sidebar.markdown("---")
        st.sidebar.subheader("📚 RAG Configuration")
        
        # Initialize session state
        if "config_saved" not in st.session_state:
            st.session_state.config_saved = False
        if "config_message" not in st.session_state:
            st.session_state.config_message = ""
        if "rag_messages" not in st.session_state:
            st.session_state.rag_messages = []
        
        # Embedding Provider Selection
        provider = st.sidebar.selectbox(
            "Embedding Provider:",
            ["Ollama", "HuggingFace"],
            index=0,
            help="Choose the provider for document embeddings"
        )
        
        # Embedding Model Selection
        if provider == "Ollama":
            try:
                models_response = requests.get(f"{FASTAPI_URL}/models")
                available_embedding_models = [m for m in models_response.json().get("models", []) if "embed" in m.lower()]
                if not available_embedding_models:
                    available_embedding_models = models_response.json().get("models", [])
            except:
                available_embedding_models = ["nomic-embed-text"]
            
            embedding_model = st.sidebar.selectbox(
                "Ollama Embedding Model:",
                available_embedding_models,
                index=0 if available_embedding_models else None,
                help="Select an embedding model from your Ollama installation"
            )
        else:  # HuggingFace
            embedding_model = st.sidebar.text_input(
                "HuggingFace Model Repo:",
                value="sentence-transformers/all-MiniLM-L6-v2",
                help="Enter a HuggingFace model repository path"
            )
        
        # Document Processing Parameters
        chunk_size = st.sidebar.number_input(
            "Chunk Size:",
            value=500,
            min_value=100,
            max_value=2000,
            step=100,
            help="Size of text chunks for document indexing (applies to standard processing)"
        )
        
        # Query Parameters
        n_results = st.sidebar.number_input(
            "Number of Results:",
            value=3,
            min_value=1,
            max_value=10,
            help="Number of relevant chunks to retrieve during querying"
        )
        
        use_multi_agent = st.sidebar.checkbox(
            "Use Multi-Agent Orchestration (CrewAI)",
            value=False,
            help="Enable advanced multi-agent RAG processing"
        )
        
        # Save Configuration Button
        if st.sidebar.button("💾 Save Configuration", type="primary"):
            with st.spinner("Saving configuration..."):
                success, message = save_rag_configuration(provider, embedding_model, chunk_size)
                if success:
                    st.session_state.config_saved = True
                    st.session_state.config_message = "✅ Configuration saved successfully!"
                else:
                    st.session_state.config_saved = False
                    st.session_state.config_message = f"❌ Failed to save configuration: {message}"
        
        # Hybrid Search Configuration
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 Search Configuration")
        
        use_hybrid_search = st.sidebar.checkbox(
            "🔍 Enable Hybrid Search (BM25 + Vector)", 
            value=False,
            help="Combines keyword-based BM25 search with semantic vector search",
            disabled=not st.session_state.config_saved
        )
        
        if use_hybrid_search:
            st.sidebar.info("💡 Hybrid search improves accuracy for complex queries")
        elif not st.session_state.config_saved:
            st.sidebar.warning("⚠️ Save configuration first")
        
        # Clear RAG chat button
        if st.sidebar.button("🗑️ Clear RAG History", type="secondary"):
            st.session_state.rag_messages = []
            st.rerun()
        
        # Display configuration status
        if st.session_state.config_message:
            if st.session_state.config_saved:
                st.success(st.session_state.config_message)
            else:
                st.error(st.session_state.config_message)
        
        # Base Model Selection
        try:
            models_response_rag = requests.get(f"{FASTAPI_URL}/models")
            available_models_rag = models_response_rag.json().get("models", ["default"])
        except:
            available_models_rag = ["default"]
        
        selected_model_rag = st.selectbox(
            "Select Base Model:",
            available_models_rag,
            help="Choose the language model for generating responses"
        )
        
        # Enhanced File Upload Section
        st.subheader("📁 Upload Documents")
        
        if not st.session_state.config_saved:
            st.warning("⚠️ Please save your configuration first before uploading documents.")
        
        # Get file type information
        file_info = get_file_type_info()
        
        # File uploader with enhanced support
        uploaded_files = st.file_uploader(
            "Upload Documents for Enhanced Processing",
            type=file_info["supported_extensions"],
            accept_multiple_files=True,
            help="Optionally upload CSVs, images, or other documents for enhanced multimodal processing",
            disabled=not st.session_state.config_saved
        )
        
        # Display file information with enhanced processing indicators
        if uploaded_files:
            
            csv_files = []
            image_files = []
            standard_files = []
            
            for file in uploaded_files:
                file_ext = file.name.split('.')[-1].lower()
                file_size = len(file.getvalue()) / 1024
                size_str = f"{file_size/1024:.1f} MB" if file_size > 1024 else f"{file_size:.1f} KB"
                
                if file_ext in ['csv', 'tsv']:
                    csv_files.append((file.name, size_str))
                elif file_ext in ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp']:
                    image_files.append((file.name, size_str))
                else:
                    standard_files.append((file.name, size_str))
        
        # Enhanced upload button
        if uploaded_files and st.button("📤 Upload & Process with Enhanced Extraction", type="primary", disabled=not st.session_state.config_saved):
            with st.spinner(f"Processing {len(uploaded_files)} files with enhanced extraction..."):
                success, results = upload_files_to_fastapi_enhanced(uploaded_files, chunk_size)
                
                if success and results:
                    st.success("✅ Enhanced processing completed!")
                    display_upload_results(results)
                    
                    # Clear cached messages for new knowledge
                    if "rag_messages" in st.session_state:
                        st.session_state.rag_messages.clear()
                        st.info("💡 Chat history cleared to reflect new knowledge base")
                    
                    logger.info("Enhanced file processing completed successfully")
                else:
                    st.error("❌ Enhanced processing failed! Check error details above.")
        
        # System Prompt Configuration
        st.subheader("⚙️ System Configuration")
        system_prompt = st.text_area(
            "System Prompt:",
            value="You are a helpful assistant with access to enhanced multimodal document knowledge. Use the detailed analysis from CSV files, extracted text from images, and comprehensive document content to provide accurate answers. If the answer requires information not in the context, clearly state that.",
            height=120,
            help="System prompt optimized for enhanced multimodal processing"
        )
        
        # Enhanced Query Section
        st.subheader("🔍 Query Your Enhanced Knowledge Base")
        
        # Display RAG chat history
        for message in st.session_state.rag_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input for RAG query
        rag_input = st.chat_input("Ask about your documents, data analysis, image content, or any uploaded materials...")
        
        if rag_input:
            # Add user message to history
            st.session_state.rag_messages.append({"role": "user", "content": rag_input})
            
            # Display user message immediately
            with st.chat_message("user"):
                st.write(rag_input)
            
            # Prepare messages for backend
            messages = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.rag_messages[:-1]]
            
            with st.chat_message("assistant"):
                search_msg = "Searching enhanced knowledge base"
                if use_hybrid_search:
                    search_msg += " (hybrid search enabled)"
                if use_multi_agent:
                    search_msg += " with multi-agent orchestration"
                search_msg += "..."

                # Create a placeholder for incremental updates
                out = st.empty()
                with st.spinner(search_msg):
                    response_text = stream_fastapi_rag_query(
                        query=rag_input,
                        messages=messages,
                        model=selected_model_rag,
                        system_prompt=system_prompt,
                        n_results=n_results,
                        use_multi_agent=use_multi_agent,
                        use_hybrid_search=use_hybrid_search,
                        placeholder=out,
                    )

                if response_text:
                    # Final render to ensure last chunk shows
                    out.markdown(_unescape_text(response_text))
                    st.session_state.rag_messages.append({"role": "assistant", "content": response_text})
                else:
                    error_msg = "❌ No response received. Please check your query and try again."
                    st.error(error_msg)
                    st.session_state.rag_messages.append({"role": "assistant", "content": error_msg})
    
    else:
        st.info("📡 Enhanced RAG functionality requires FastAPI backend. Please start the backend server.")
     
# Footer status
st.sidebar.markdown("---")
if backend_available:
    if capabilities and capabilities.get("status") == "ready":
        st.sidebar.write("🟢 Enhanced processing ready")
    else:
        st.sidebar.write("🟡 Basic functionality available")
else:
    st.sidebar.write("🔴 Limited functionality (backend offline)")