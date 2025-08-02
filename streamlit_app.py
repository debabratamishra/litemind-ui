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

async def call_local_ollama(messages, model: str = "default", temperature: float = 0.7):
    """Fallback to local Ollama service"""
    response = ""
    async for chunk in stream_ollama(messages, model=model, temperature=temperature):
        response += chunk
    return response

def call_fastapi_rag_query(query: str, model: str, system_prompt: str, chunk_size: int = 500, n_results: int = 3):
    """Call FastAPI RAG query endpoint"""
    try:
        response = requests.post(
            f"{FASTAPI_URL}/api/rag/query",
            json={
                "query": query,
                "model": model,
                "system_prompt": system_prompt,
                "chunk_size": chunk_size,
                "n_results": n_results
            },
            timeout=120
        )
        return response.text if response.status_code == 200 else None
    except Exception as e:
        st.error(f"RAG API Error: {str(e)}")
        return None

def upload_file_to_fastapi(file):
    """Upload a single file to FastAPI backend"""
    try:
        files = {"files": (file.name, file.getvalue(), file.type)}  # Note: Endpoint expects 'files' key for list
        response = requests.post(f"{FASTAPI_URL}/api/rag/upload", files=files)
        return response.status_code == 200
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
st.sidebar.title("Configuration")
st.sidebar.write(f"**Backend Mode:** {backend_mode}")

# Main application logic
tab1, tab2 = st.tabs(["💬 Chat", "📚 RAG"])

with tab1:
    st.header("Chat Interface")
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

    # Chat input
    user_input = st.chat_input("Enter your message:")
    if user_input:
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
    if backend_available:
        try:
            models_response_rag = requests.get(f"{FASTAPI_URL}/models")
            available_models_rag = models_response_rag.json().get("models", ["default"])
        except:
            available_models_rag = ["default"]
        selected_model_rag = st.selectbox("Select Base Model:", available_models_rag)

        # File upload (support multiple)
        uploaded_files = st.file_uploader("Upload Documents", type=['pdf', 'txt', 'docx'], accept_multiple_files=True)
        if uploaded_files and st.button("Upload"):
            success = True
            for uploaded_file in uploaded_files:
                if not upload_file_to_fastapi(uploaded_file):
                    success = False
            if success:
                st.success("All files uploaded successfully!")
            else:
                st.error("Some uploads failed!")

        # RAG Query
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.number_input("Chunk Size:", value=500, min_value=100)
        with col2:
            n_results = st.number_input("Number of Results:", value=3, min_value=1)

        system_prompt = st.text_area("System Prompt:",
                                     value="You are a helpful assistant. Answer based on the document context only. If the answer is not in the context, say you don’t know.")
        rag_query = st.text_area("Enter your query:")
        if st.button("Query Documents") and rag_query:
            with st.spinner("Querying documents..."):
                response = call_fastapi_rag_query(rag_query, selected_model_rag, system_prompt, chunk_size, n_results)
                if response:
                    st.write("**Response:**")
                    st.write(response)
    else:
        st.info("RAG functionality requires FastAPI backend. Please start the backend server.")

# Status footer
st.sidebar.markdown("---")
st.sidebar.write("**System Status:**")
st.sidebar.write(f"Backend: {backend_mode}")
if backend_available:
    st.sidebar.write("🟢 All systems operational")
else:
    st.sidebar.write("🟡 Limited functionality")
