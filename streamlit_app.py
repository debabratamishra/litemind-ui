"""
LiteMindUI - A production-ready Streamlit interface for Large Language Models.

This application provides an intuitive web interface for interacting with LLMs
through Ollama models.
It also supports document Q&A through RAG (Retrieval-Augmented Generation).
"""
import logging
import streamlit as st
from pathlib import Path
import os
from importlib import import_module

from app.frontend.services.backend_service import backend_service

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
            st.session_state.selected_page = "Chat"
        
        # Validate selected page is valid
        valid_pages = ["Chat", "RAG"]
        if st.session_state.selected_page not in valid_pages:
            st.session_state.selected_page = "Chat"
            
        st.session_state.setdefault("chat_messages", [])
    
    def render_sidebar_header(self):
        st.sidebar.markdown("# 🤖 LiteMindUI")
        st.sidebar.markdown("---")
        
        self.render_backend_selection()
        self.render_page_navigation()
        self.render_system_status()
    
    def render_backend_selection(self):
        st.sidebar.subheader("Backend Provider")

        if st.session_state.backend_available:
            st.sidebar.info("🦙 Ollama Backend")
            st.session_state.current_backend = "ollama"
        else:
            st.sidebar.info("🔴 FastAPI backend required for Ollama support")
    
    def render_page_navigation(self):
        st.sidebar.markdown("---")
        
        # Get current page index, defaulting to Chat (0) if not found
        page_options = ["Chat", "RAG"]
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
        st.sidebar.subheader("System Status")
        
        is_docker = st.session_state.get("is_docker_deployment", False)
        
        if st.session_state.backend_available:
            st.sidebar.success("✅ FastAPI Backend Connected")
            st.sidebar.info("🦙 Ollama Mode Active")
        else:
            st.sidebar.warning("⚠️ Using Local Backend")

    def _render_page_module(self, module_name: str, render_name: str, class_name: str):
        """Load a page module lazily and render it using the exported helper or class."""
        module = import_module(module_name)

        render_fn = getattr(module, render_name, None)
        if callable(render_fn):
            render_fn()
            return

        page_class = getattr(module, class_name, None)
        if page_class is None:
            raise ImportError(
                f"{module_name} does not export {render_name} or {class_name}"
            )

        page = page_class()
        if hasattr(page, "render_sidebar_config"):
            page.render_sidebar_config()
        if hasattr(page, "render"):
            page.render()
            return

        raise AttributeError(f"{class_name} in {module_name} does not implement render()")
    
    def run(self):
        self.render_sidebar_header()
        
        # Ensure we have a valid page selected
        selected_page = st.session_state.get("selected_page", "Chat")
        
        # Route to the appropriate page
        try:
            if selected_page == "Chat":
                self._render_page_module(
                    "app.frontend.pages.chat_page",
                    "render_chat_page",
                    "ChatPage",
                )
            elif selected_page == "RAG":
                self._render_page_module(
                    "app.frontend.pages.rag_page",
                    "render_rag_page",
                    "RAGPage",
                )
            else:
                # Fallback to chat if invalid page
                st.session_state.selected_page = "Chat"
                self._render_page_module(
                    "app.frontend.pages.chat_page",
                    "render_chat_page",
                    "ChatPage",
                )
        except Exception as e:
            # Error handling for page rendering
            st.error(f"Error loading {selected_page}: {str(e)}")
            st.info("Please try refreshing the page or selecting a different page.")


def main():
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()
