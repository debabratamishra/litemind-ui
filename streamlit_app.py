# """
# Streamlit front-end for the LLM WebUI
# ------------------------------------
# Run it with:
#     streamlit run streamlit_app.py
# """

# from __future__ import annotations

# import asyncio
# from pathlib import Path

# import nest_asyncio
# import streamlit as st

# from app.services.ollama import stream_ollama
# from app.services.rag_service import RAGService, CrewAIRAGOrchestrator
# from streamlit.components.v1 import html

# # one-line injection; height=0 keeps it invisible
# html("""
# <link rel="manifest" href="/pwa/manifest.webmanifest">
# <script>
# if ('serviceWorker' in navigator) {
#   window.addEventListener('load', () => {
#     navigator.serviceWorker.register('/pwa/sw.js').catch(()=>{});
#   });
# }
# </script>
# """, height=0)

# nest_asyncio.apply()  # allow nested event-loops inside Streamlit callbacks


# def run_async(coro):
#     """Run an async coroutine from Streamlit’s sync context."""
#     return asyncio.run(coro)

# st.set_page_config(
#     page_title="LLM WebUI (Streamlit)",
#     page_icon="🪄",
#     layout="wide",
# )

# st.sidebar.title("🧭 Navigation")
# page = st.sidebar.radio("Choose a mode", ["Chat", "Document QA (RAG)"])

# # one RAG client across reruns
# @st.cache_resource(show_spinner=False)
# def _get_rag():
#     return RAGService()

# rag_service = _get_rag()
# orchestrator = CrewAIRAGOrchestrator(rag_service)

# # chat history persists across reruns
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []           # [{role, content}, …]

# # ---------------------------------------------------------------------------
# # CHAT
# # ---------------------------------------------------------------------------
# if page == "Chat":
#     st.header("💬 Chat")

#     for m in st.session_state.chat_history:
#         with st.chat_message(m["role"]):
#             st.markdown(m["content"])

#     if prompt := st.chat_input("Ask me anything …"):
#         # log & render the user’s turn
#         st.session_state.chat_history.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         # assistant placeholder
#         with st.chat_message("assistant"):
#             placeholder = st.empty()
#             placeholder.markdown("_thinking…_")

#         # collect streamed chunks from Ollama
#         async def _answer(p: str) -> str:
#             chunks: list[str] = []
#             async for chunk in stream_ollama([{"role": "user", "content": p}]):
#                 chunks.append(chunk)
#                 placeholder.markdown("".join(chunks) + "▌")
#             return "".join(chunks)

#         answer = run_async(_answer(prompt))
#         placeholder.markdown(answer)
#         st.session_state.chat_history.append({"role": "assistant", "content": answer})

# # ---------------------------------------------------------------------------
# # DOCUMENT QA / RAG
# # ---------------------------------------------------------------------------
# else:
#     st.header("📄 Document QA (RAG)")

#     # upload & ingest PDFs
#     pdf = st.file_uploader("Upload a PDF to index", type=["pdf"])
#     if pdf is not None:
#         path = Path("uploads") / pdf.name
#         path.parent.mkdir(exist_ok=True)
#         path.write_bytes(pdf.getvalue())
#         with st.spinner("Indexing document …"):
#             run_async(rag_service.add_document(str(path), pdf.name))
#         st.success("Document added! You can ask questions below.")

#     # query controls
#     sys_prompt = st.text_area(
#         "System prompt (optional)",
#         value=(
#             "You are a helpful assistant.  Answer *only* from the context. "
#             "If the answer is not in the context, say you don’t know."
#         ),
#         height=70,
#     )
#     query = st.text_area("Question about your documents", height=100)
    

#     if st.button("Ask") and query.strip():
#         placeholder = st.empty()
#         placeholder.markdown("_thinking…_")

#         async def _rag(q: str) -> str:
#             chunks: list[str] = []
#             async for ch in orchestrator.query(q, system_prompt=sys_prompt, n_results=3):
#                 chunks.append(ch)
#                 placeholder.markdown("".join(chunks) + "▌")
#             return "".join(chunks)

#         answer = run_async(_rag(query))
#         placeholder.markdown(answer)


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
from app.services.rag_service import RAGService, CrewAIRAGOrchestrator
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

def call_fastapi_chat(message: str, model: str = "default"):
    """Call FastAPI chat endpoint"""
    try:
        response = requests.post(
            f"{FASTAPI_URL}/api/chat",
            json={"message": message, "model": model},
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

async def call_local_ollama(messages):
    """Fallback to local Ollama service"""
    response = ""
    async for chunk in stream_ollama(messages):
        response += chunk
    return response

def call_fastapi_rag_query(query: str, system_prompt: str, chunk_size: int = 500, n_results: int = 3):
    """Call FastAPI RAG query endpoint"""
    try:
        response = requests.post(
            f"{FASTAPI_URL}/api/rag/query",
            json={
                "query": query,
                "system_prompt": system_prompt,
                "chunk_size": chunk_size,
                "n_results": n_results
            },
            timeout=60
        )
        return response.text if response.status_code == 200 else None
    except Exception as e:
        st.error(f"RAG API Error: {str(e)}")
        return None

def upload_file_to_fastapi(file):
    """Upload file to FastAPI backend"""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
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
        available_models = ["local-ollama"]
    
    selected_model = st.selectbox("Select Model:", available_models)
    
    # Chat input
    user_input = st.chat_input("Enter your message:")
    
    if user_input:
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        
        # Process with appropriate backend
        with st.spinner("Processing..."):
            if backend_available:
                response = call_fastapi_chat(user_input, selected_model)
                if response:
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
            else:
                # Fallback to local processing
                messages = [{"role": "user", "content": user_input}]
                response = asyncio.run(call_local_ollama(messages))
                st.session_state.chat_messages.append({"role": "assistant", "content": response})
    
    # Display chat history
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

with tab2:
    st.header("RAG Interface")
    
    if backend_available:
        # File upload
        uploaded_file = st.file_uploader("Upload Document", type=['pdf', 'txt', 'docx'])
        if uploaded_file and st.button("Upload"):
            if upload_file_to_fastapi(uploaded_file):
                st.success("File uploaded successfully!")
            else:
                st.error("Upload failed!")
        
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
                response = call_fastapi_rag_query(rag_query, system_prompt, chunk_size, n_results)
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
