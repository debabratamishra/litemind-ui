"""
LiteMindUI - A production-ready Streamlit interface for Large Language Models.

This application provides an intuitive web interface for interacting with LLMs
through both Ollama (local models) and vLLM (HuggingFace models) backends.
It also supports document Q&A through RAG (Retrieval-Augmented Generation).
"""
import logging
import streamlit as st
from pathlib import Path
import os

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
    
    @staticmethod
    def _detect_docker_environment() -> bool:
        """Detect if running inside a Docker container."""
        # Check for common container indicators
        container_indicators = [
            Path('/.dockerenv').exists(),
            Path('/proc/1/cgroup').exists() and 'docker' in Path('/proc/1/cgroup').read_text(errors='ignore'),
            os.getenv('CONTAINER') is not None,
            os.getenv('DOCKER_CONTAINER') is not None
        ]
        return any(container_indicators)
        
    def setup_page_config(self):
        st.set_page_config(
            page_title="LiteMindUI", 
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
        
        # Check if running in Docker
        if "is_docker_deployment" not in st.session_state:
            st.session_state.is_docker_deployment = self._detect_docker_environment()
        
        # Initialize page selection with explicit default
        if "selected_page" not in st.session_state:
            st.session_state.selected_page = "💬 Chat"
        
        # Validate selected page is valid
        valid_pages = ["💬 Chat", "📚 RAG"]
        if st.session_state.selected_page not in valid_pages:
            st.session_state.selected_page = "💬 Chat"
            
        st.session_state.setdefault("chat_messages", [])
    
    def render_sidebar_header(self):
        st.sidebar.markdown("# 🤖 LiteMindUI")
        st.sidebar.markdown("---")
        
        self.render_backend_selection()
        self.render_page_navigation()
        self.render_system_status()
    
    def render_backend_selection(self):
        st.sidebar.subheader("⚙️ Backend Provider")
        
        # Check if running in Docker environment
        is_docker = st.session_state.get("is_docker_deployment", False)
        
        if st.session_state.backend_available:
            # If Docker deployment, only show Ollama option
            if is_docker:
                st.sidebar.info("🦙 Ollama Backend")
                st.sidebar.warning("⚠️ vLLM is not supported with Docker installation yet. It will be added shortly.")
                backend_provider = "ollama"
                # Force current backend to ollama in Docker
                st.session_state.current_backend = "ollama"
            else:
                # Native deployment, show both options
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
        
        # Show vLLM configuration if selected and not in Docker
        if (st.session_state.backend_available and 
            st.session_state.current_backend == "vllm" and
            not is_docker):
            st.sidebar.markdown("---")
            setup_vllm_backend()
    
    def render_page_navigation(self):
        st.sidebar.markdown("---")
        
        # Get current page index, defaulting to Chat (0) if not found
        page_options = ["💬 Chat", "📚 RAG"]
        try:
            current_index = page_options.index(st.session_state.selected_page)
        except (ValueError, KeyError):
            current_index = 0
            st.session_state.selected_page = page_options[0]
        
        # Use a callback to handle page changes immediately
        def on_page_change():
            if st.session_state.page_selector_new != st.session_state.selected_page:
                st.session_state.selected_page = st.session_state.page_selector_new
        
        page = st.sidebar.selectbox(
            "Navigate to:", 
            page_options,
            index=current_index,
            key="page_selector_new",
            on_change=on_page_change
        )
    
    def render_system_status(self):
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔧 System Status")
        
        is_docker = st.session_state.get("is_docker_deployment", False)
        
        if st.session_state.backend_available:
            st.sidebar.success("✅ FastAPI Backend Connected")
            backend_provider = st.session_state.get("current_backend", "ollama")
            if backend_provider == "vllm" and not is_docker:
                st.sidebar.info("⚡ vLLM Mode Active")
            else:
                st.sidebar.info("🦙 Ollama Mode Active")
        else:
            st.sidebar.warning("⚠️ Using Local Backend")
    
    def run(self):
        self.render_sidebar_header()
        
        # Ensure we have a valid page selected
        selected_page = st.session_state.get("selected_page", "💬 Chat")
        
        # Route to the appropriate page
        try:
            if selected_page == "💬 Chat":
                render_chat_page()
            elif selected_page == "📚 RAG":
                render_rag_page()
            else:
                # Fallback to chat if invalid page
                st.session_state.selected_page = "💬 Chat"
                render_chat_page()
        except Exception as e:
            # Error handling for page rendering
            st.error(f"Error loading {selected_page}: {str(e)}")
            st.info("Please try refreshing the page or selecting a different page.")


def main():
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()
