"""
Shared UI Components Module.

This module provides reusable Streamlit UI components to eliminate
code duplication between chat_page.py and rag_page.py.
"""
import os
import logging
import streamlit as st
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

OPENROUTER_DEFAULT_API_BASE = (
    os.getenv("OPENROUTER_API_BASE")
    or os.getenv("OPENROUTER_BASE_URL")
    or "https://openrouter.ai/api/v1"
)
OPENROUTER_DEFAULT_MODEL = os.getenv("DEFAULT_OPENROUTER_MODEL", "openai/gpt-4o-mini")


# =============================================================================
# Data Classes for Configuration
# =============================================================================

@dataclass
class GenerationConfig:
    """Configuration for LLM generation parameters."""
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "repetition_penalty": self.repetition_penalty,
        }


@dataclass
class MemoryStats:
    """Statistics about the current memory state."""
    total_messages: int
    total_tokens: int
    summary_tokens: int
    has_summary: bool
    max_tokens: int
    usage_percentage: float
    needs_summarization: bool


# =============================================================================
# Shared UI Rendering Functions
# =============================================================================

def render_memory_indicator(stats: MemoryStats) -> None:
    """
    Render a visual memory usage indicator.
    
    Args:
        stats: MemoryStats object containing current memory state
    """
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


def render_generation_settings(
    prefix: str,
    expanded: bool = True
) -> GenerationConfig:
    """
    Render generation settings sliders in an expander.
    
    Args:
        prefix: Prefix for session state keys (e.g., "chat" or "rag")
        expanded: Whether the expander should be expanded by default
        
    Returns:
        GenerationConfig: Current configuration values
    """
    with st.sidebar.expander("Generation Settings", expanded=expanded):
        # Temperature slider
        temperature = st.slider(
            "Temperature:", 
            0.0, 1.0, 
            st.session_state.get(f"{prefix}_temperature", 0.7), 
            0.1,
            help="Controls randomness in responses. Lower = more focused, higher = more creative"
        )
        st.session_state[f"{prefix}_temperature"] = temperature
        
        # Max tokens slider — upper bound lifted to 16 384 for GenUI artifacts
        max_tokens = st.slider(
            "Max Tokens:", 
            256, 16384, 
            st.session_state.get(f"{prefix}_max_tokens", 2048), 
            256,
            help="Maximum tokens to generate. GenUI mode auto-raises this to 16 384 at the backend so complex apps generate fully."
        )
        st.session_state[f"{prefix}_max_tokens"] = max_tokens
        
        # Top P (nucleus sampling)
        top_p = st.slider(
            "Top P (Nucleus Sampling):",
            0.0, 1.0,
            st.session_state.get(f"{prefix}_top_p", 0.9),
            0.05,
            help="Controls diversity via nucleus sampling. Lower = more focused, higher = more diverse"
        )
        st.session_state[f"{prefix}_top_p"] = top_p
        
        # Frequency penalty
        frequency_penalty = st.slider(
            "Frequency Penalty:",
            -2.0, 2.0,
            st.session_state.get(f"{prefix}_frequency_penalty", 0.0),
            0.1,
            help="Penalize tokens based on their frequency in the text. Positive = less repetition"
        )
        st.session_state[f"{prefix}_frequency_penalty"] = frequency_penalty
        
        # Repetition penalty
        repetition_penalty = st.slider(
            "Repetition Penalty:",
            0.0, 2.0,
            st.session_state.get(f"{prefix}_repetition_penalty", 1.0),
            0.1,
            help="Penalize repeated tokens. Values > 1.0 reduce repetition"
        )
        st.session_state[f"{prefix}_repetition_penalty"] = repetition_penalty
    
    return GenerationConfig(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        repetition_penalty=repetition_penalty,
    )


def render_reasoning_config(key_prefix: str = "") -> None:
    """
    Render reasoning display configuration.
    
    Args:
        key_prefix: Optional prefix for widget keys to avoid conflicts
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("Reasoning Display")
    
    expand_key = f"{key_prefix}_reasoning_expanded" if key_prefix else None
    hide_key = f"{key_prefix}_hide_reasoning" if key_prefix else None
    
    st.session_state.show_reasoning_expanded = st.sidebar.checkbox(
        "Expand reasoning by default",
        value=st.session_state.get("show_reasoning_expanded", False),
        help="Show model reasoning sections expanded by default",
        key=expand_key
    )
    
    hide_reasoning = st.sidebar.checkbox(
        "Hide reasoning completely",
        value=st.session_state.get("hide_reasoning", False),
        help="Completely hide reasoning sections from responses",
        key=hide_key
    )
    st.session_state.hide_reasoning = hide_reasoning


def render_memory_config(
    memory_manager: Any,
    prefix: str,
    history_key: str = "history_enabled",
    memory_key: str = "memory_enabled"
) -> None:
    """
    Render conversation memory configuration in sidebar.
    
    Args:
        memory_manager: The memory manager instance to get stats from
        prefix: Prefix for session state keys (e.g., "chat" or "rag")
        history_key: Suffix for history enabled key
        memory_key: Suffix for memory enabled key
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("Conversation Settings")
    
    full_history_key = f"{prefix}_{history_key}"
    full_memory_key = f"{prefix}_{memory_key}"
    
    # History persistence toggle
    history_enabled = st.sidebar.checkbox(
        "Save conversation history",
        value=st.session_state.get(full_history_key, True),
        help="Persist conversations for later access"
    )
    st.session_state[full_history_key] = history_enabled
    
    # Memory toggle
    memory_enabled = st.sidebar.checkbox(
        "Enable context memory",
        value=st.session_state.get(full_memory_key, True),
        help="Remember context from earlier in the conversation"
    )
    st.session_state[full_memory_key] = memory_enabled
    
    if memory_enabled:
        # Display memory stats
        stats = memory_manager.get_stats()
        
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


def render_backend_selector() -> str:
    """Render the inference backend selector and provider-level settings."""
    options = {"Ollama": "ollama", "OpenRouter": "openrouter"}
    current_backend = st.session_state.get("current_backend", "ollama")
    labels = list(options.keys())
    default_index = next(
        (index for index, label in enumerate(labels) if options[label] == current_backend),
        0,
    )

    selected_label = st.sidebar.selectbox(
        "Inference Backend",
        labels,
        index=default_index,
        key="current_backend_label",
        help="Choose which LiteLLM-supported backend powers chat and RAG generation.",
    )
    backend = options[selected_label]
    st.session_state["current_backend"] = backend

    if backend == "openrouter":
        st.session_state.setdefault("openrouter_api_key", "")
        st.session_state.setdefault("openrouter_api_base", OPENROUTER_DEFAULT_API_BASE)
        st.sidebar.caption(
            "OpenRouter model ids can be entered as provider/model. The app prefixes openrouter/ automatically."
        )
        st.sidebar.text_input(
            "OpenRouter API Base",
            key="openrouter_api_base",
            help="Defaults to https://openrouter.ai/api/v1",
        )
        st.sidebar.text_input(
            "OpenRouter API Key",
            key="openrouter_api_key",
            type="password",
            help="Optional if the FastAPI backend already has OPENROUTER_API_KEY configured.",
        )

    return backend


def get_default_openrouter_model() -> str:
    """Return the default OpenRouter model shown in the UI."""
    return OPENROUTER_DEFAULT_MODEL


def get_backend_request_config(backend_provider: str) -> Dict[str, Optional[str]]:
    """Return backend-specific request settings from session state."""
    config: Dict[str, Optional[str]] = {
        "backend": backend_provider,
        "api_base": None,
        "api_key": None,
    }

    if backend_provider == "openrouter":
        api_base = (st.session_state.get("openrouter_api_base") or "").strip()
        api_key = (st.session_state.get("openrouter_api_key") or "").strip()
        config["api_base"] = api_base or None
        config["api_key"] = api_key or None

    return config


def validate_backend_setup(backend_provider: str) -> bool:
    """
    Validate backend setup for the current provider.
    
    Args:
        backend_provider: The backend provider ("ollama")
        
    Returns:
        bool: True if setup is valid, False otherwise
    """
    backend_available = bool(st.session_state.get("backend_available", False))
    if not backend_available:
        st.error("❌ FastAPI backend is required. Please start the backend server.")
        return False

    if backend_provider == "ollama":
        return True

    if backend_provider == "openrouter":
        return True

    st.error("❌ Unsupported backend selected.")
    return False


def create_simple_summary(
    messages: List[Dict[str, str]],
    existing_summary: Optional[str] = None
) -> str:
    """
    Create a simple extractive summary of messages.
    
    This is a shared implementation used by both chat and RAG pages.
    In production, this could call the LLM for a better summary.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        existing_summary: Existing summary to incorporate
        
    Returns:
        str: Combined summary text
    """
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


def get_generation_config_from_session(prefix: str) -> Dict[str, Any]:
    """
    Get generation configuration from session state.
    
    Args:
        prefix: Prefix for session state keys (e.g., "chat" or "rag")
        
    Returns:
        Dict with generation parameters
    """
    return {
        "temperature": st.session_state.get(f"{prefix}_temperature", 0.7),
        "max_tokens": st.session_state.get(f"{prefix}_max_tokens", 2048),
        "top_p": st.session_state.get(f"{prefix}_top_p", 0.9),
        "frequency_penalty": st.session_state.get(f"{prefix}_frequency_penalty", 0.0),
        "repetition_penalty": st.session_state.get(f"{prefix}_repetition_penalty", 1.0),
    }
