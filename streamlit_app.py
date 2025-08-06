"""
Streamlit front-end for the LLM WebUI with FastAPI backend detection
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
import io  # For streaming text updates
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

def call_fastapi_rag_query(query: str, messages: list, model: str, system_prompt: str, n_results: int = 3, use_multi_agent: bool = False):
    """Call FastAPI RAG query endpoint with full message history"""
    try:
        response = requests.post(
            f"{FASTAPI_URL}/api/rag/query",
            json={
                "query": query,
                "messages": messages,
                "model": model,
                "system_prompt": system_prompt,
                "n_results": n_results,
                "use_multi_agent": use_multi_agent
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

# New function for multi-file upload with chunk_size
def upload_files_to_fastapi(uploaded_files, chunk_size: int = 500):
    """Upload multiple files to FastAPI backend with chunk size configuration"""
    if not uploaded_files:
        return False
    try:
        files = [("files", (file.name, file.getvalue(), file.type)) for file in uploaded_files]
        # Include chunk_size in the upload request
        response = requests.post(
            f"{FASTAPI_URL}/api/rag/upload", 
            files=files,
            data={"chunk_size": chunk_size}  # NEW: Include chunk_size for indexing
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
st.set_page_config(page_title="LLM WebUI", layout="wide")

# Check backend status
backend_available = check_fastapi_backend()
if backend_available:
    st.success("✅ FastAPI Backend Connected")
    backend_mode = "FastAPI"
else:
    st.warning("⚠️ FastAPI Backend Unavailable - Using Local Backend")
    backend_mode = "Local"

st.title("🤖 LLM WebUI")

# Main application logic
tab1, tab2 = st.tabs(["💬 Chat", "📚 RAG"])

with tab1:
    st.header("Chat Interface")
    # Sidebar configuration for Chat
    st.sidebar.title("Configuration")
    st.sidebar.write(f"**Backend Mode:** {backend_mode}")
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    # Model selection
    if backend_available:
        try:
            models_response = requests.get(f"{FASTAPI_URL}/models")
            available_models = models_response.json().get("models", ["default"])
        except:
            available_models = ["default"]
    else:
        available_models = ["gemma3n:e2b"]  # Fallback to local model
    selected_model = st.selectbox("Select Model:", available_models)
    # Temperature slider
    temperature = st.slider("Temperature (0.0 = deterministic, 1.0 = creative):", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    logger.info(f"Selected model: {selected_model}, Temperature: {temperature}")
    # Chat input
    user_input = st.chat_input("Enter your message:")
    if user_input:
        logger.info(f"User input: {user_input}")
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        # Process with appropriate backend
        with st.spinner("Processing..."):
            if backend_available:
                response = call_fastapi_chat(user_input, selected_model, temperature)
                if response:
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
            else:
                # Fallback to local processing
                messages = [{"role": "user", "content": user_input}]
                response = asyncio.run(call_local_ollama(messages, model=selected_model, temperature=temperature))
                st.session_state.chat_messages.append({"role": "assistant", "content": response})
    # Display chat history
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

with tab2:
    st.header("RAG Interface")
    # ============ SIDEBAR CONFIGURATION ============
    st.sidebar.header("📊 RAG Configuration")
    
    # Initialize session state for configuration status
    if "config_saved" not in st.session_state:
        st.session_state.config_saved = False
    if "config_message" not in st.session_state:
        st.session_state.config_message = ""
    
    # MODIFIED: Initialize session state for RAG messages
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []
    
    # Embedding Provider Selection
    provider = st.sidebar.selectbox(
        "Embedding Provider:",
        ["Ollama", "HuggingFace"],
        index=0,  # Default to Ollama
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
            available_embedding_models = ["nomic-embed-text"]  # Default fallback
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
        help="Size of text chunks for document indexing (affects future uploads)"
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
        help="Enable advanced multi-agent RAG processing (may be slower but more accurate)"
    )
    
    # NEW: Save Configuration Button
    if st.sidebar.button("💾 Save Configuration", type="primary"):
        with st.spinner("Saving configuration..."):
            success, message = save_rag_configuration(provider, embedding_model, chunk_size)
            if success:
                st.session_state.config_saved = True
                st.session_state.config_message = "✅ Configuration saved successfully! You can now upload documents with these settings."
            else:
                st.session_state.config_saved = False
                st.session_state.config_message = f"❌ Failed to save configuration: {message}"
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Tip**: Save your configuration before uploading documents. Chunk size affects how documents are processed.")
    
    # ============ CENTER PANEL CONTENT ============
    if backend_available:
        # Display configuration status in center panel
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
            disabled=not st.session_state.config_saved  # Disable if config not saved
        )
        
        if uploaded_files and st.button("📤 Upload & Index Documents", type="primary", disabled=not st.session_state.config_saved):
            with st.spinner("Uploading and indexing files..."):
                if upload_files_to_fastapi(uploaded_files, chunk_size):
                    st.success("✅ All files uploaded and indexed successfully!")
                    logger.info("Files uploaded successfully for RAG processing")
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
        rag_input = st.chat_input("Ask a context constrained question...")

        if rag_input:
            # Add user message to history
            st.session_state.rag_messages.append({"role": "user", "content": rag_input})
            
            # Prepare messages for backend (full history)
            messages = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.rag_messages]
            
            with st.spinner("Thinking..."):
                response = call_fastapi_rag_query(
                    query=rag_input,  # Still pass the latest query for retrieval
                    messages=messages,  # NEW: Pass full history for chaining
                    model=selected_model_rag,
                    system_prompt=system_prompt,
                    n_results=n_results,
                    use_multi_agent=use_multi_agent
                )
                if response:
                    # Add assistant response to history
                    st.session_state.rag_messages.append({"role": "assistant", "content": response})
                    
                    st.rerun()  # Force refresh to show new message
                else:
                    st.error("❌ No response received. Please check your query and try again.")

    else:
        st.info("📡 RAG functionality requires FastAPI backend. Please start the backend server.")

# Status footer in sidebar
st.sidebar.markdown("---")
st.sidebar.write("**System Status:**")
st.sidebar.write(f"Backend: {backend_mode}")
if backend_available:
    st.sidebar.write("🟢 All systems operational")
else:
    st.sidebar.write("🟡 Limited functionality")
