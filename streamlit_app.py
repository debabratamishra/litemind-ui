"""
LLM WebUI - A production-ready Streamlit interface for Large Language Models.
Supports multiple backends (Ollama, vLLM) with voice input and RAG capabilities.
"""
import logging
import streamlit as st

from app.frontend.services.backend_service import backend_service
from app.frontend.components.vllm_config import setup_vllm_backend
from app.frontend.pages.chat_page import render_chat_page
from app.frontend.pages.rag_page import render_rag_page

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamlitApp:
    """Main application controller"""
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
        
    def setup_page_config(self):
        st.set_page_config(
            page_title="LLM WebUI", 
            layout="wide", 
            initial_sidebar_state="expanded"
        )
    
    def initialize_session_state(self):
        """Initialize required session state variables"""
        if "backend_available" not in st.session_state:
            st.session_state.backend_available = backend_service.check_health()
        
        if "capabilities" not in st.session_state:
            st.session_state.capabilities = (
                backend_service.get_processing_capabilities() 
                if st.session_state.backend_available else None
            )
        
        if "selected_page" not in st.session_state:
            st.session_state.selected_page = "💬 Chat"
            
        st.session_state.setdefault("chat_messages", [])
    
    def render_sidebar_header(self):
        st.sidebar.title("🤖 LLM WebUI")
        st.sidebar.markdown("---")
        
        self.render_backend_selection()
        self.render_page_navigation()
        self.render_system_status()
    
    def render_backend_selection(self):
        st.sidebar.subheader("⚙️ Backend Provider")
        
        if st.session_state.backend_available:
            backend_provider = st.sidebar.radio(
                "Select LLM Backend:",
                ["ollama", "vllm"],
                format_func=lambda x: "🦙 Ollama" if x == "ollama" else "⚡ vLLM",
                key="backend_provider",
                help="Choose between Ollama (local) or vLLM (Huggingface) backend"
            )
            
            # Handle backend switching
            if "current_backend" not in st.session_state:
                st.session_state.current_backend = backend_provider
            elif st.session_state.current_backend != backend_provider:
                st.session_state.current_backend = backend_provider
                if "vllm_model" in st.session_state:
                    del st.session_state.vllm_model
                st.rerun()
        else:
            backend_provider = "ollama"
            st.sidebar.info("🔴 FastAPI backend required for vLLM support")
        
        # Show vLLM configuration if selected
        if (st.session_state.backend_available and 
            st.session_state.current_backend == "vllm"):
            st.sidebar.markdown("---")
            setup_vllm_backend()
    
    def render_page_navigation(self):
        st.sidebar.markdown("---")
        
        page = st.sidebar.selectbox(
            "Navigate to:", 
            ["💬 Chat", "📚 RAG"], 
            index=0 if st.session_state.selected_page == "💬 Chat" else 1,
            key="page_selector"
        )
        
        if page != st.session_state.selected_page:
            st.session_state.selected_page = page
    
    def render_system_status(self):
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔧 System Status")
        
        if st.session_state.backend_available:
            st.sidebar.success("✅ FastAPI Backend Connected")
            backend_provider = st.session_state.get("current_backend", "ollama")
            if backend_provider == "vllm":
                st.sidebar.info("⚡ vLLM Mode Active")
            else:
                st.sidebar.info("🦙 Ollama Mode Active")
        else:
            st.sidebar.warning("⚠️ Using Local Backend")
    
    def run(self):
        self.render_sidebar_header()
        
        selected_page = st.session_state.selected_page
        
        if selected_page == "💬 Chat":
            render_chat_page()
        elif selected_page == "📚 RAG":
            render_rag_page()


def main():
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()
