"""
Chat interface for LLM conversations with voice input support and conversation memory.
"""
import logging
import streamlit as st
from typing import Dict, List, Optional

from ..components.voice_input import get_voice_input
from ..components.text_renderer import render_llm_text, render_plain_text, render_web_search_text
from ..components.streaming_handler import streaming_handler
from ..components.web_search_toggle import WebSearchToggle
from ..components.tts_player import render_tts_button
from ..components.conversation_sidebar import get_chat_sidebar
from ..services.backend_service import backend_service
from ..utils.memory_manager import ChatMemoryManager

logger = logging.getLogger(__name__)


class ChatPage:
    """Chat interface controller with conversation memory support"""
    
    def __init__(self):
        self.backend_available = st.session_state.get("backend_available", False)
        self.web_search_toggle = WebSearchToggle()
        self.memory_manager = ChatMemoryManager()
        self.conversation_sidebar = get_chat_sidebar()
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state variables for web search and memory"""
        if "web_search_enabled" not in st.session_state:
            st.session_state.web_search_enabled = False
        
        if "serp_api_token_status" not in st.session_state:
            st.session_state.serp_api_token_status = None
        
        # Initialize memory-related session state
        if "chat_memory_enabled" not in st.session_state:
            st.session_state.chat_memory_enabled = True
        
        # Initialize conversation history enabled state
        if "chat_history_enabled" not in st.session_state:
            st.session_state.chat_history_enabled = True
    
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
    
    def _get_conversation_context(self) -> tuple[List[Dict[str, str]], Optional[str]]:
        """
        Get conversation history and summary for API calls.
        
        Returns:
            Tuple of (conversation_history, conversation_summary)
        """
        if not st.session_state.get("chat_memory_enabled", True):
            return [], None
        
        # Get history excluding the most recent user message (which we're about to send)
        history = self.memory_manager.get_history_for_api(exclude_last=1)
        summary = self.memory_manager.summary
        
        return history, summary
        
    def render(self):
        realtime_active = st.session_state.get("realtime_voice_mode_chat", False)

        if not realtime_active:
            st.title("LLM Chat Interface")
            
            # Display memory indicator if memory is enabled
            if st.session_state.get("chat_memory_enabled", True):
                stats = self.memory_manager.get_stats()
                if stats.total_messages > 0:
                    self._render_memory_indicator(stats)
        
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
    
    def _render_memory_indicator(self, stats):
        """Render a visual memory usage indicator."""
        # Choose color based on usage
        if stats.usage_percentage < 50:
            color = "#4CAF50"  # green
        elif stats.usage_percentage < 75:
            color = "#FF9800"  # orange
        else:
            color = "#f44336"  # red
        
        summary_indicator = "📝" if stats.has_summary else ""
        
        st.markdown(
            f"""<div style="font-size: 0.75em; color: #888; padding: 4px 0;">
            <span style="color: {color};">●</span> 
            Context: {stats.usage_percentage:.0f}% ({stats.total_messages} messages) {summary_indicator}
            </div>""",
            unsafe_allow_html=True
        )
    
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
        
        # Get conversation context BEFORE adding the new message
        conversation_history, conversation_summary = self._get_conversation_context()
        
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        
        # Save user message to conversation history if enabled
        if st.session_state.get("chat_history_enabled", True):
            self.conversation_sidebar.save_message("user", user_input)
        
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
                        max_tokens=config["max_tokens"],
                        top_p=config["top_p"],
                        frequency_penalty=config["frequency_penalty"],
                        repetition_penalty=config["repetition_penalty"],
                        backend=backend_provider,
                        hf_token=config.get("hf_token"),
                        placeholder=out,
                        use_fastapi=self.backend_available,
                        conversation_history=conversation_history,
                        conversation_summary=conversation_summary,
                        session_id=self.memory_manager.session_id
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
                        max_tokens=config["max_tokens"],
                        top_p=config["top_p"],
                        frequency_penalty=config["frequency_penalty"],
                        repetition_penalty=config["repetition_penalty"],
                        backend=backend_provider,
                        hf_token=config.get("hf_token"),
                        placeholder=out,
                        use_fastapi=self.backend_available,
                        conversation_history=conversation_history,
                        conversation_summary=conversation_summary,
                        session_id=self.memory_manager.session_id
                    )
        
        if reply:
            assistant_message = {"role": "assistant", "content": reply}
            if use_web_search:
                assistant_message["format"] = "web_search"
            st.session_state.chat_messages.append(assistant_message)
            
            # Save assistant message to conversation history if enabled
            if st.session_state.get("chat_history_enabled", True):
                metadata = {"format": "web_search"} if use_web_search else None
                self.conversation_sidebar.save_message("assistant", reply, metadata)
            
            # Check if we need to trigger summarization
            self._check_and_trigger_summarization()
            
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
            "max_tokens": st.session_state.get("chat_max_tokens", 2048),
            "top_p": st.session_state.get("chat_top_p", 0.9),
            "frequency_penalty": st.session_state.get("chat_frequency_penalty", 0.0),
            "repetition_penalty": st.session_state.get("chat_repetition_penalty", 1.0),
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
        # Render conversation history sidebar first
        if st.session_state.get("chat_history_enabled", True):
            self.conversation_sidebar.render()
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Chat Configuration")
        
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
        
        # Generation settings in expander
        with st.sidebar.expander("Generation Settings", expanded=True):
            # Temperature slider
            temperature = st.slider(
                "Temperature:", 
                0.0, 1.0, 
                st.session_state.get("chat_temperature", 0.7), 
                0.1,
                help="Controls randomness in responses. Lower = more focused, higher = more creative"
            )
            st.session_state.chat_temperature = temperature
            
            # Max tokens slider
            max_tokens = st.slider(
                "Max Tokens:", 
                256, 8192, 
                st.session_state.get("chat_max_tokens", 2048), 
                256,
                help="Maximum number of tokens to generate in the response"
            )
            st.session_state.chat_max_tokens = max_tokens
            
            # Top P (nucleus sampling)
            top_p = st.slider(
                "Top P (Nucleus Sampling):",
                0.0, 1.0,
                st.session_state.get("chat_top_p", 0.9),
                0.05,
                help="Controls diversity via nucleus sampling. Lower = more focused, higher = more diverse"
            )
            st.session_state.chat_top_p = top_p
            
            # Frequency penalty
            frequency_penalty = st.slider(
                "Frequency Penalty:",
                -2.0, 2.0,
                st.session_state.get("chat_frequency_penalty", 0.0),
                0.1,
                help="Penalize tokens based on their frequency in the text. Positive = less repetition"
            )
            st.session_state.chat_frequency_penalty = frequency_penalty
            
            # Repetition penalty
            repetition_penalty = st.slider(
                "Repetition Penalty:",
                0.0, 2.0,
                st.session_state.get("chat_repetition_penalty", 1.0),
                0.1,
                help="Penalize repeated tokens. Values > 1.0 reduce repetition"
            )
            st.session_state.chat_repetition_penalty = repetition_penalty
        
        # Reasoning settings
        self._render_reasoning_config()
        
        # Clear chat (clears current session, not history)
        if st.sidebar.button("🗑️ Clear Current Chat", type="secondary"):
            st.session_state.chat_messages.clear()
            # Clear active conversation to start fresh
            self.conversation_sidebar.active_conversation_id = None
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
            available_models = ["gemma3:1b"]
        
        selected_model = st.sidebar.selectbox(
            "Select Model:", 
            available_models,
            key="selected_chat_model"
        )
    
    def _render_reasoning_config(self):
        st.sidebar.markdown("---")
        st.sidebar.subheader("Reasoning Display")
        
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
        
        # Memory configuration
        self._render_memory_config()
    
    def _render_memory_config(self):
        """Render conversation memory configuration in sidebar."""
        st.sidebar.markdown("---")
        st.sidebar.subheader("Conversation Settings")
        
        # History persistence toggle
        history_enabled = st.sidebar.checkbox(
            "Save conversation history",
            value=st.session_state.get("chat_history_enabled", True),
            help="Persist conversations for later access"
        )
        st.session_state.chat_history_enabled = history_enabled
        
        # Memory toggle
        memory_enabled = st.sidebar.checkbox(
            "Enable context memory",
            value=st.session_state.get("chat_memory_enabled", True),
            help="Remember context from earlier in the conversation"
        )
        st.session_state.chat_memory_enabled = memory_enabled
        
        if memory_enabled:
            # Display memory stats
            stats = self.memory_manager.get_stats()
            
            # Progress bar for context usage
            usage_label = f"Context: {stats.usage_percentage:.0f}%"
            st.sidebar.progress(min(stats.usage_percentage / 100, 1.0), text=usage_label)
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                st.sidebar.caption(f"📝 {stats.total_messages} msgs")
            with col2:
                st.sidebar.caption(f"🎯 ~{stats.total_tokens} tokens")
            
            if stats.has_summary:
                st.sidebar.success("📋 Conversation summarized", icon="✅")
            
            if stats.needs_summarization:
                st.sidebar.warning("Context near limit - will summarize soon")
    
    def _check_and_trigger_summarization(self):
        """Check if summarization is needed and trigger it."""
        if not st.session_state.get("chat_memory_enabled", True):
            return
        
        stats = self.memory_manager.get_stats()
        
        if stats.needs_summarization:
            logger.info("Context limit approaching, triggering summarization...")
            
            # Get messages to summarize
            messages_to_summarize = self.memory_manager.prune_for_summarization()
            
            if messages_to_summarize:
                # Format for summary
                summary_text = self.memory_manager.format_messages_for_summary_prompt(
                    messages_to_summarize,
                    self.memory_manager.summary
                )
                
                simple_summary = self._create_simple_summary(messages_to_summarize)
                
                self.memory_manager.set_summary(simple_summary)
                logger.info(f"Created summary with {len(simple_summary)} characters")
    
    def _create_simple_summary(self, messages: List[Dict[str, str]]) -> str:
        """
        Create a simple extractive summary of messages.
        
        In production, this would call the LLM for a better summary,
        but we use a simple approach for immediate functionality.
        """
        existing_summary = self.memory_manager.summary
        
        # Extract key points from messages
        summary_parts = []
        
        if existing_summary:
            summary_parts.append(f"Previous context: {existing_summary[:500]}")
        
        # Summarize user queries and key assistant responses
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "")
            
            # Truncate long messages
            if len(content) > 200:
                content = content[:200] + "..."
            
            if role == "user":
                summary_parts.append(f"User asked about: {content}")
            elif role == "assistant":
                # Take first sentence or first 100 chars
                first_sentence = content.split('.')[0] if '.' in content else content[:100]
                summary_parts.append(f"Assistant explained: {first_sentence}")
        
        # Combine and limit total length
        combined = " | ".join(summary_parts)
        
        # Limit to ~2000 characters (roughly 500 tokens)
        if len(combined) > 2000:
            combined = combined[:2000] + "..."
        
        return combined


def render_chat_page():
    chat_page = ChatPage()
    chat_page.render_sidebar_config()
    chat_page.render()
