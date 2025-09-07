"""
Chat interface for LLM conversations with voice input support and tool use capabilities.
"""
import asyncio
import logging
import streamlit as st
from typing import Dict, List

from ..components.voice_input import get_voice_input
from ..components.text_renderer import render_llm_text
from ..components.streaming_handler import streaming_handler
from ..services.backend_service import backend_service
from app.services.tool_use_chat import tool_use_chat_service

logger = logging.getLogger(__name__)


class ChatPage:
    """Chat interface controller"""
    
    def __init__(self):
        self.backend_available = st.session_state.get("backend_available", False)
        
    def render(self):
        st.title("💬 LLM Chat Interface")
        
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        
        backend_provider = st.session_state.get("current_backend", "ollama")
        
        # Tool use toggle
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("**Enhanced Chat with Tool Use Capabilities**")
        with col2:
            tool_use_enabled = st.checkbox(
                "🔧 Enable Tools", 
                value=st.session_state.get("tool_use_enabled", False),
                help="Enable tool use capabilities (web search, file operations, etc.)"
            )
            st.session_state.tool_use_enabled = tool_use_enabled
        
        if tool_use_enabled:
            # Show available tools
            with st.expander("🧰 Available Tools"):
                tools = tool_use_chat_service.get_available_tools()
                if tools:
                    for tool_name, tool_info in tools.items():
                        st.markdown(f"**{tool_info['name']}**: {tool_info['description']}")
                else:
                    st.info("No tools available. Please check n8n service status.")
        
        self._display_chat_history()
        
        user_input = get_voice_input("Enter your message...", "chat")
        
        if user_input:
            if tool_use_enabled:
                self._process_user_input_with_tools(user_input, backend_provider)
            else:
                self._process_user_input(user_input, backend_provider)
    
    def _display_chat_history(self):
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
    
    def _process_user_input(self, user_input: str, backend_provider: str):
        # Validate setup
        if not self._validate_setup(backend_provider):
            return
        
        config = self._get_chat_config(backend_provider)
        
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            render_llm_text(user_input)
        
        # Generate response
        with st.chat_message("assistant"):
            out = st.empty()
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
            st.session_state.chat_messages.append({"role": "assistant", "content": reply})
            # Use rerun sparingly - only when necessary for UI updates
            st.rerun()
        else:
            error_msg = "❌ No response received."
            st.error(error_msg)
            st.session_state.chat_messages.append({"role": "assistant", "content": "No response."})
                    # Don't rerun on error - let user try again naturally
    
    def _process_user_input_with_tools(self, user_input: str, backend_provider: str):
        """Process user input with tool use capabilities"""
        # Validate setup
        if not self._validate_setup(backend_provider):
            return
        
        config = self._get_chat_config(backend_provider)
        
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            render_llm_text(user_input)
        
        # Generate response with tool use
        with st.chat_message("assistant"):
            out = st.empty()
            status_text = self._get_status_text(backend_provider, config.get("model", "default"))
            
            with st.spinner(status_text):
                # Create conversation with tool use capabilities
                conversation = tool_use_chat_service.create_tool_use_conversation(
                    user_input, 
                    st.session_state.chat_messages[:-1]  # Exclude the just-added user message
                )
                
                # Generate initial response
                initial_response = ""
                
                # Stream the LLM response
                for chunk in streaming_handler._stream_response_generator(
                    messages=conversation,
                    model=config["model"], 
                    temperature=config["temperature"],
                    backend=backend_provider,
                    hf_token=config.get("hf_token"),
                    use_fastapi=self.backend_available
                ):
                    initial_response += chunk
                    out.markdown(initial_response + "▌")
                
                out.markdown(initial_response)
                
                # Check if there are tool calls in the response
                if "<tool_call>" in initial_response:
                    with st.spinner("🔧 Executing tools..."):
                        # Process tool calls
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            processed_response, tool_results = loop.run_until_complete(
                                tool_use_chat_service.process_message_with_tools(initial_response)
                            )
                        finally:
                            loop.close()
                        
                        # Update the display with processed response
                        out.markdown(processed_response)
                        
                        # Store message with tool results
                        st.session_state.chat_messages.append({
                            "role": "assistant", 
                            "content": processed_response,
                            "tool_results": tool_results
                        })
                        
                        # Show tool execution details
                        if tool_results:
                            with st.expander("🔧 Tool Execution Details"):
                                for i, result in enumerate(tool_results, 1):
                                    col1, col2 = st.columns([1, 3])
                                    with col1:
                                        if result.get('success'):
                                            st.success(f"✅ Tool {i}")
                                        else:
                                            st.error(f"❌ Tool {i}")
                                        st.markdown(f"**{result.get('tool', 'Unknown')}**")
                                    with col2:
                                        if result.get('success'):
                                            st.json(result.get('result', {}))
                                        else:
                                            st.error(result.get('error', 'Unknown error'))
                else:
                    # No tool calls, just store the regular response
                    st.session_state.chat_messages.append({
                        "role": "assistant", 
                        "content": initial_response
                    })
            
            # Trigger rerun for UI updates
            st.rerun()
    
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
    
    def _get_status_text(self, backend_provider: str, model: str) -> str:
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
