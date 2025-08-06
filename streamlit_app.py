"""
Streamlit front-end for the LLM WebUI with FastAPI backend detection
Converted to multi-page application with separate Chat and RAG pages
"""

from __future__ import annotations
import asyncio
import requests
from pathlib import Path
import nest_asyncio
import streamlit as st
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
FASTAPI_TIMEOUT = 10  # seconds

def check_fastapi_backend():
    """Check if FastAPI backend is available"""
    try:
        response = requests.get(f"{FASTAPI_URL}/health", timeout=FASTAPI_TIMEOUT)
        return response.status_code == 200
    except (requests.exceptions.RequestException, requests.exceptions.Timeout):
        return False

def call_fastapi_chat(message: str, model: str = "default", temperature: float = 0.7):
    """Call FastAPI chat endpoint"""
    try:
        response = requests.post(
            f"{FASTAPI_URL}/api/chat",
            json={"message": message, "model": model, "temperature": temperature},
            timeout=30
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
                "use_hybrid_search": use_hybrid_search  # New parameter
            },
            timeout=120
        )
        return response.text if response.status_code == 200 else None
    except Exception as e:
        st.error(f"RAG API Error: {str(e)}")
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

def upload_files_to_fastapi(uploaded_files, chunk_size: int = 500):
    """Upload multiple files to FastAPI backend with chunk size configuration"""
    if not uploaded_files:
        return False
    
    try:
        files = [("files", (file.name, file.getvalue(), file.type)) for file in uploaded_files]
        response = requests.post(
            f"{FASTAPI_URL}/api/rag/upload",
            files=files,
            data={"chunk_size": chunk_size}
        )
        
        if response.status_code == 200:
            return True
        else:
            st.error(f"Upload failed with status {response.status_code}: {response.text}")
            return False
    except Exception as e:
        st.error(f"Upload Error: {str(e)}")
        return False

# Initialize Streamlit app
st.set_page_config(
    page_title="LLM WebUI", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check backend status (shared across pages)
if 'backend_available' not in st.session_state:
    st.session_state.backend_available = check_fastapi_backend()

backend_available = st.session_state.backend_available
backend_mode = "FastAPI" if backend_available else "Local"

# Page navigation
st.sidebar.title("🤖 LLM WebUI")
st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Navigate to:",
    ["💬 Chat", "📚 RAG"],
    index=0
)

# Backend status in sidebar (shared)
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 System Status")
if backend_available:
    st.sidebar.success("✅ FastAPI Backend Connected")
else:
    st.sidebar.warning("⚠️ Using Local Backend")

st.sidebar.write(f"**Mode:** {backend_mode}")

# ============ CHAT PAGE ============
if page == "💬 Chat":
    st.title("💬 LLM Chat Interface")

    # ---------- Sidebar configuration ----------
    st.sidebar.markdown("---")
    st.sidebar.subheader("💬 Chat Configuration")

    # Initialise session storage
    st.session_state.setdefault("chat_messages", [])

    # Available models
    if backend_available:
        try:
            available_models = requests.get(f"{FASTAPI_URL}/models").json().get("models", ["default"])
        except Exception:
            available_models = ["default"]
    else:
        available_models = ["gemma3n:e2b"]  # Local fallback

    selected_model = st.sidebar.selectbox("Select Model:", available_models)
    temperature = st.sidebar.slider("Temperature:", 0.0, 1.0, 0.7, 0.1)

    if st.sidebar.button("🗑️ Clear Chat History", type="secondary"):
        st.session_state.chat_messages.clear()
        st.rerun()

    logger.info("Model: %s | Temperature: %.1f", selected_model, temperature)

    # ---------- Conversation area ----------
    if st.session_state.chat_messages:
        # Display existing conversation
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
    else:
        st.markdown(
            """
            <div style='text-align:center; color:gray; font-size:1.2em; padding-top:16rem; padding-bottom:4rem;'>
                How can I help with today?
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------- Input bar at the bottom ----------
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
    st.title("📚 LLM RAG Interface")
    
    if backend_available:
        # RAG-specific sidebar configuration
        st.sidebar.markdown("---")
        st.sidebar.subheader("📚 RAG Configuration")
        
        # Initialize session state for configuration status
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
            help="Size of text chunks for document indexing"
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
        
        # Hybrid Search Configuration (Under Save Configuration Button)
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 Search Configuration")
        
        # NEW: Hybrid Search Option
        use_hybrid_search = st.sidebar.checkbox(
            "🔍 Enable Hybrid Search (BM25 + Vector)", 
            value=False,
            help="Combines keyword-based BM25 search with semantic vector search for better retrieval accuracy",
            disabled=not st.session_state.config_saved
        )
        
        if use_hybrid_search:
            st.sidebar.info("💡 Hybrid search combines exact keyword matching with semantic similarity for improved results")
        elif not st.session_state.config_saved:
            st.sidebar.warning("⚠️ Save configuration first to enable hybrid search")
        
        # Clear RAG chat button
        if st.sidebar.button("🗑️ Clear RAG History", type="secondary"):
            st.session_state.rag_messages = []
            st.rerun()
        
        st.sidebar.markdown("---")
        st.sidebar.info("💡 **Tip**: Save your configuration before uploading documents.")
        
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
        
        logger.info(f"Selected RAG model: {selected_model_rag}")
        
        # File Upload Section
        st.subheader("📁 Document Upload")
        if not st.session_state.config_saved:
            st.warning("⚠️ Please save your configuration first before uploading documents.")
        
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            help="Upload PDF, TXT, or DOCX files for RAG processing",
            disabled=not st.session_state.config_saved
        )
        
        if uploaded_files and st.button("📤 Upload & Index Documents", type="primary", disabled=not st.session_state.config_saved):
            with st.spinner("Uploading and indexing files..."):
                if upload_files_to_fastapi(uploaded_files, chunk_size):
                    st.success("✅ All files uploaded and indexed successfully!")
                    logger.info("Files uploaded successfully for RAG processing")
                    # Clear any cached messages to reflect new knowledge
                    if "rag_messages" in st.session_state:
                        st.session_state.rag_messages.clear()
                else:
                    st.error("❌ Upload failed! Check the error details above.")
        
        # System Prompt Configuration
        st.subheader("⚙️ System Configuration")
        system_prompt = st.text_area(
            "System Prompt:",
            value="You are a helpful assistant. Answer based on the document context only. If the answer is not in the context, say you don't know.",
            height=100,
            help="Define how the AI should behave when answering questions"
        )
        
        # Query Section
        st.subheader("🔍 Query Documents")
        
        # Display RAG chat history
        for message in st.session_state.rag_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input for RAG query
        rag_input = st.chat_input("Ask a question about your documents...")
        
        if rag_input:
            # Add user message to history
            st.session_state.rag_messages.append({"role": "user", "content": rag_input})
            
            # Display user message immediately
            with st.chat_message("user"):
                st.write(rag_input)
            
            # Prepare messages for backend (excluding the current message)
            messages = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.rag_messages[:-1]]
            
            with st.chat_message("assistant"):
                with st.spinner("Searching knowledge base..."):
                    response = call_fastapi_rag_query(
                        query=rag_input,
                        messages=messages,
                        model=selected_model_rag,
                        system_prompt=system_prompt,
                        n_results=n_results,
                        use_multi_agent=use_multi_agent,
                        use_hybrid_search=use_hybrid_search  # Pass hybrid search parameter
                    )
                    
                    if response:
                        st.write(response)
                        # Add assistant response to history
                        st.session_state.rag_messages.append({"role": "assistant", "content": response})
                    else:
                        error_msg = "❌ No response received. Please check your query and try again."
                        st.error(error_msg)
                        st.session_state.rag_messages.append({"role": "assistant", "content": error_msg})
    
    else:
        st.info("📡 RAG functionality requires FastAPI backend. Please start the backend server.")
        st.sidebar.markdown("---")
        st.sidebar.info("RAG configuration unavailable without FastAPI backend")

# Footer status (shared)
st.sidebar.markdown("---")
if backend_available:
    st.sidebar.write("🟢 All systems operational")
else:
    st.sidebar.write("🟡 Limited functionality")
