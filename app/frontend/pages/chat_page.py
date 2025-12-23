"""
Chat interface for LLM conversations with voice input support.
"""
import logging
import streamlit as st
from typing import Dict

from ..components.voice_input import get_voice_input
from ..components.text_renderer import render_llm_text, render_plain_text, render_web_search_text
from ..components.streaming_handler import streaming_handler
from ..components.web_search_toggle import WebSearchToggle
from ..components.tts_player import render_tts_button
from ..services.backend_service import backend_service

logger = logging.getLogger(__name__)


class ChatPage:
    """Chat interface controller"""
    
    def __init__(self):
        self.backend_available = st.session_state.get("backend_available", False)
        self.web_search_toggle = WebSearchToggle()
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state variables for web search"""
        if "web_search_enabled" not in st.session_state:
            st.session_state.web_search_enabled = False
        
        if "serp_api_token_status" not in st.session_state:
            st.session_state.serp_api_token_status = None
    
    def _render_web_search_toggle(self) -> bool:
        """
        Render web search toggle in prompt area.
        
        Returns:
            bool: Current toggle state
        """
        return self.web_search_toggle.render()
    
    def _get_web_search_status(self) -> Dict[str, bool]:
        """
        Check if web search is enabled and token is valid.
        
        Returns:
            dict: Dictionary with 'enabled' and 'token_valid' keys
        """
        web_search_enabled = st.session_state.get("web_search_enabled", False)
        
        if not web_search_enabled:
            return {"enabled": False, "token_valid": False}
        
        # Get token status
        token_status = st.session_state.get("serp_api_token_status")
        if token_status is None:
            token_status = self.web_search_toggle.get_token_status()
            st.session_state.serp_api_token_status = token_status
        
        token_valid = token_status.get("status") == "valid"
        
        return {"enabled": True, "token_valid": token_valid}
        
    def render(self):
        realtime_active = st.session_state.get("realtime_voice_mode_chat", False)

        if not realtime_active:
            st.title("💬 LLM Chat Interface")
        
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        
        backend_provider = st.session_state.get("current_backend", "ollama")
        
        if not realtime_active:
            self._display_chat_history()
            
            # Render web search toggle before voice input
            self._render_web_search_toggle()
        
        user_input = get_voice_input("Enter your message...", "chat")
        
        if user_input:
            self._process_user_input(user_input, backend_provider)
    
    def _display_chat_history(self):
        if st.session_state.chat_messages:
            for idx, msg in enumerate(st.session_state.chat_messages):
                with st.chat_message(msg["role"]):
                    msg_format = msg.get("format", "")
                    if msg_format == "web_search":
                        render_web_search_text(msg["content"])
                    elif msg_format == "plain":
                        render_plain_text(msg["content"])
                    else:
                        render_llm_text(msg["content"])
                    
                    # Add TTS play button for assistant messages (always show)
                    if msg["role"] == "assistant":
                        render_tts_button(msg["content"], "chat", idx)
        else:
            st.markdown(
                """
                <div style='text-align:center; color:gray; font-size:1.2em; padding-top:16rem; padding-bottom:4rem;'>
                    How can I help you today?
                </div>
                """,
                unsafe_allow_html=True,
            )
    
    def _process_user_input(self, user_input: str, backend_provider: str):
        # Validate setup
        if not self._validate_setup(backend_provider):
            return
        
        config = self._get_chat_config(backend_provider)
        
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            render_llm_text(user_input)
        
        # Check web search status
        web_search_status = self._get_web_search_status()
        use_web_search = web_search_status["enabled"] and web_search_status["token_valid"]
        
        # Generate response
        with st.chat_message("assistant"):
            out = st.empty()
            
            # Route to web search endpoint if enabled and token valid
            if use_web_search:
                status_text = "Searching web..."
                
                with st.spinner(status_text):
                    reply = streaming_handler.stream_web_search_response(
                        message=user_input,
                        model=config["model"],
                        temperature=config["temperature"],
                        backend=backend_provider,
                        hf_token=config.get("hf_token"),
                        placeholder=out,
                        use_fastapi=self.backend_available
                    )
            else:
                # Display error message if web search enabled but token invalid
                if web_search_status["enabled"] and not web_search_status["token_valid"]:
                    st.warning("⚠️ SerpAPI token is required to perform Web search. Defaulting to local results")
                
                status_text = self._get_status_text(backend_provider, config.get("model", "default"))
                
                with st.spinner(status_text):
                    reply = streaming_handler.stream_chat_response(
                        message=user_input,
                        model=config["model"],
                        temperature=config["temperature"],
                        backend=backend_provider,
                        hf_token=config.get("hf_token"),
                        placeholder=out,
                        use_fastapi=self.backend_available
                    )
        
        if reply:
            assistant_message = {"role": "assistant", "content": reply}
            if use_web_search:
                assistant_message["format"] = "web_search"
            st.session_state.chat_messages.append(assistant_message)
            # Use rerun sparingly - only when necessary for UI updates
            st.rerun()
        else:
            error_msg = "❌ No response received."
            st.error(error_msg)
            st.session_state.chat_messages.append({"role": "assistant", "content": "No response."})
            # Don't rerun on error - let user try again naturally
    
    def _validate_setup(self, backend_provider: str) -> bool:
        is_docker = st.session_state.get("is_docker_deployment", False)
        
        # Prevent vLLM usage in Docker
        if backend_provider == "vllm" and is_docker:
            st.error("❌ vLLM is not supported with Docker installation yet. Please use Ollama backend.")
            return False
        
        if backend_provider == "vllm":
            vllm_model = st.session_state.get("vllm_model")
            if not vllm_model:
                st.error("❌ Please configure and load a vLLM model first")
                return False
        return True
    
    def _get_chat_config(self, backend_provider: str) -> Dict:
        config = {
            "temperature": st.session_state.get("chat_temperature", 0.7),
            "hf_token": st.session_state.get("hf_token") if backend_provider == "vllm" else None
        }
        
        if backend_provider == "vllm":
            config["model"] = st.session_state.get("vllm_model", "no-model")
        else:
            config["model"] = st.session_state.get("selected_chat_model", "default")
        
        return config
    
    def _get_status_text(self, backend_provider: str, model: str, web_search_active: bool = False) -> str:
        """
        Get status text for the spinner.
        
        Args:
            backend_provider: The backend provider (ollama, vllm)
            model: The model name
            web_search_active: Whether web search is active
            
        Returns:
            str: Status text to display
        """
        if web_search_active:
            return "Searching web..."
        
        if backend_provider == "vllm":
            return f"Thinking (vLLM - {model})..."
        return "Thinking..."
    
    def render_sidebar_config(self):
        st.sidebar.markdown("---")
        st.sidebar.subheader("💬 Chat Configuration")
        
        backend_provider = st.session_state.get("current_backend", "ollama")
        is_docker = st.session_state.get("is_docker_deployment", False)
        
        # Model selection - only show if not vLLM in Docker
        if backend_provider == "vllm" and is_docker:
            st.sidebar.warning("⚠️ vLLM not supported in Docker deployment")
            st.sidebar.info("Please use Ollama backend instead")
        elif backend_provider == "vllm":
            self._render_vllm_model_config()
        else:
            self._render_ollama_model_config()
        
        # Temperature slider
        temperature = st.sidebar.slider("Temperature:", 0.0, 1.0, 0.7, 0.1)
        st.session_state.chat_temperature = temperature
        
        # Reasoning settings
        self._render_reasoning_config()
        
        # Clear chat
        if st.sidebar.button("🗑️ Clear Chat History", type="secondary"):
            st.session_state.chat_messages.clear()
            # Clear any other chat-related state that might interfere
            if "last_user_input" in st.session_state:
                del st.session_state.last_user_input
            st.rerun()
    
    def _render_vllm_model_config(self):
        vllm_model = st.session_state.get("vllm_model")
        if vllm_model:
            st.sidebar.success(f"🎯 Active Model: {vllm_model}")
        else:
            st.sidebar.warning("⚠️ No vLLM model loaded")
            st.sidebar.info("👆 Configure vLLM above to load a model")
    
    def _render_ollama_model_config(self):
        if self.backend_available:
            available_models = backend_service.get_available_models()
        else:
            available_models = ["gemma3n:e2b"]
        
        selected_model = st.sidebar.selectbox(
            "Select Model:", 
            available_models,
            key="selected_chat_model"
        )
    
    def _render_reasoning_config(self):
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


def render_chat_page():
    chat_page = ChatPage()
    chat_page.render_sidebar_config()
    chat_page.render()
