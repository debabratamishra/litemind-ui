"""
Frontend Conversation Memory Manager for Streamlit.

This module provides client-side conversation memory management that:
- Tracks conversation history in Streamlit session state
- Estimates token counts
- Automatically triggers summarization when needed
- Provides the appropriate context for API calls
"""
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import hashlib
from datetime import datetime

import streamlit as st

logger = logging.getLogger(__name__)

# Token estimation constants
CHARS_PER_TOKEN = 4  # Conservative estimate
DEFAULT_MAX_CONTEXT_TOKENS = 24000  # Leave headroom for 32K context
SUMMARIZE_THRESHOLD = 0.70  # Trigger summarization at 70% capacity
KEEP_RECENT_MESSAGES = 4  # Always keep last N messages


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


def generate_session_id(prefix: str = "session") -> str:
    """Generate a unique session ID."""
    timestamp = datetime.now().isoformat()
    unique_part = hashlib.md5(f"{timestamp}{id(st)}".encode()).hexdigest()[:12]
    return f"{prefix}_{unique_part}"


def estimate_tokens(text: str) -> int:
    """Estimate token count for a text string."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


class FrontendMemoryManager:
    """
    Manages conversation memory on the frontend using Streamlit session state.
    
    This class provides:
    - Storage of conversation history in session state
    - Token estimation and tracking
    - Automatic summarization triggering
    - Context preparation for API calls
    """
    
    def __init__(
        self,
        session_key: str = "chat_messages",
        summary_key: str = "conversation_summary",
        session_id_key: str = "session_id",
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
    ):
        """
        Initialize the memory manager.
        
        Args:
            session_key: Session state key for storing messages
            summary_key: Session state key for storing summary
            session_id_key: Session state key for session ID
            max_context_tokens: Maximum tokens allowed in context
        """
        self.session_key = session_key
        self.summary_key = summary_key
        self.session_id_key = session_id_key
        self.max_context_tokens = max_context_tokens
        
        # Initialize session state if needed
        self._ensure_initialized()
    
    def _ensure_initialized(self):
        """Ensure session state is properly initialized."""
        if self.session_key not in st.session_state:
            st.session_state[self.session_key] = []
        
        if self.summary_key not in st.session_state:
            st.session_state[self.summary_key] = None
        
        if self.session_id_key not in st.session_state:
            st.session_state[self.session_id_key] = generate_session_id()
    
    @property
    def session_id(self) -> str:
        """Get the current session ID."""
        self._ensure_initialized()
        return st.session_state[self.session_id_key]
    
    @property
    def messages(self) -> List[Dict[str, str]]:
        """Get current message list."""
        self._ensure_initialized()
        return st.session_state[self.session_key]
    
    @property
    def summary(self) -> Optional[str]:
        """Get the current conversation summary."""
        self._ensure_initialized()
        return st.session_state[self.summary_key]
    
    @summary.setter
    def summary(self, value: Optional[str]):
        """Set the conversation summary."""
        st.session_state[self.summary_key] = value
    
    def add_message(self, role: str, content: str) -> Dict[str, str]:
        """
        Add a message to the conversation history.
        
        Args:
            role: Message role (user, assistant, system)
            content: Message content
            
        Returns:
            The added message dict
        """
        self._ensure_initialized()
        message = {"role": role, "content": content}
        st.session_state[self.session_key].append(message)
        return message
    
    def get_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        self._ensure_initialized()
        
        messages = st.session_state[self.session_key]
        summary = st.session_state[self.summary_key]
        
        # Calculate tokens
        total_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)
        summary_tokens = estimate_tokens(summary) if summary else 0
        combined_tokens = total_tokens + summary_tokens
        
        # Calculate percentage
        usage_percentage = (combined_tokens / self.max_context_tokens) * 100
        
        # Check if summarization is needed
        threshold_tokens = int(self.max_context_tokens * SUMMARIZE_THRESHOLD)
        needs_summarization = combined_tokens >= threshold_tokens and len(messages) > KEEP_RECENT_MESSAGES
        
        return MemoryStats(
            total_messages=len(messages),
            total_tokens=total_tokens,
            summary_tokens=summary_tokens,
            has_summary=summary is not None,
            max_tokens=self.max_context_tokens,
            usage_percentage=round(usage_percentage, 1),
            needs_summarization=needs_summarization
        )
    
    def prepare_context_for_api(
        self,
        new_message: str,
        exclude_last_n: int = 0
    ) -> Tuple[List[Dict[str, str]], Optional[str], int]:
        """
        Prepare conversation context for API call.
        
        Args:
            new_message: The new user message (not yet added to history)
            exclude_last_n: Number of recent messages to exclude (e.g., if already added)
            
        Returns:
            Tuple of (conversation_history, conversation_summary, estimated_tokens)
        """
        self._ensure_initialized()
        
        messages = st.session_state[self.session_key]
        summary = st.session_state[self.summary_key]
        
        # Get messages for context (excluding recent if specified)
        if exclude_last_n > 0:
            context_messages = messages[:-exclude_last_n] if len(messages) > exclude_last_n else []
        else:
            context_messages = messages.copy()
        
        # Calculate tokens
        total_tokens = sum(estimate_tokens(m.get("content", "")) for m in context_messages)
        total_tokens += estimate_tokens(summary) if summary else 0
        total_tokens += estimate_tokens(new_message)
        
        return context_messages, summary, total_tokens
    
    def get_history_for_api(self, exclude_last: int = 1) -> List[Dict[str, str]]:
        """
        Get conversation history formatted for API.
        
        Args:
            exclude_last: Number of recent messages to exclude
            
        Returns:
            List of message dicts with 'role' and 'content'
        """
        self._ensure_initialized()
        messages = st.session_state[self.session_key]
        
        if exclude_last > 0 and len(messages) > exclude_last:
            return [{"role": m["role"], "content": m["content"]} 
                    for m in messages[:-exclude_last]]
        elif exclude_last > 0:
            return []
        else:
            return [{"role": m["role"], "content": m["content"]} for m in messages]
    
    def prune_for_summarization(self, keep_recent: int = KEEP_RECENT_MESSAGES) -> List[Dict[str, str]]:
        """
        Prune messages for summarization, keeping only recent ones.
        
        Returns the messages that were removed (for summarization).
        
        Args:
            keep_recent: Number of recent messages to keep
            
        Returns:
            List of messages that were removed
        """
        self._ensure_initialized()
        messages = st.session_state[self.session_key]
        
        if len(messages) <= keep_recent:
            return []
        
        # Split messages
        messages_to_summarize = messages[:-keep_recent]
        messages_to_keep = messages[-keep_recent:]
        
        # Update session state
        st.session_state[self.session_key] = messages_to_keep
        
        logger.info(
            f"Pruned {len(messages_to_summarize)} messages for summarization, "
            f"keeping {len(messages_to_keep)} recent messages"
        )
        
        return messages_to_summarize
    
    def set_summary(self, summary: str):
        """Set the conversation summary after summarization."""
        st.session_state[self.summary_key] = summary
        logger.info(f"Set conversation summary ({estimate_tokens(summary)} tokens)")
    
    def clear(self):
        """Clear all conversation history and summary."""
        st.session_state[self.session_key] = []
        st.session_state[self.summary_key] = None
        st.session_state[self.session_id_key] = generate_session_id()
        logger.info("Cleared conversation memory")
    
    def format_messages_for_summary_prompt(
        self,
        messages: List[Dict[str, str]],
        existing_summary: Optional[str] = None
    ) -> str:
        """
        Format messages for a summarization prompt.
        
        Args:
            messages: Messages to summarize
            existing_summary: Existing summary to incorporate
            
        Returns:
            Formatted text for summarization
        """
        parts = []
        
        if existing_summary:
            parts.append(f"Previous context summary: {existing_summary}")
            parts.append("")
        
        parts.append("Conversation to summarize:")
        for msg in messages:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        
        return "\n".join(parts)


# Create specialized memory managers for different pages
class ChatMemoryManager(FrontendMemoryManager):
    """Memory manager for the Chat interface."""
    
    def __init__(self):
        super().__init__(
            session_key="chat_messages",
            summary_key="chat_conversation_summary",
            session_id_key="chat_session_id"
        )


class RAGMemoryManager(FrontendMemoryManager):
    """Memory manager for the RAG interface."""
    
    def __init__(self):
        super().__init__(
            session_key="rag_messages",
            summary_key="rag_conversation_summary",
            session_id_key="rag_session_id"
        )


def render_memory_indicator(memory_manager: FrontendMemoryManager):
    """
    Render a visual indicator of memory usage.
    
    Args:
        memory_manager: The memory manager to display stats for
    """
    stats = memory_manager.get_stats()
    
    # Choose color based on usage
    if stats.usage_percentage < 50:
        color = "green"
    elif stats.usage_percentage < 75:
        color = "orange"
    else:
        color = "red"
    
    # Create a compact indicator
    indicator_html = f"""
    <div style="font-size: 0.8em; color: gray; padding: 2px 8px; 
                border-radius: 4px; background: rgba(100,100,100,0.1); 
                display: inline-flex; align-items: center; gap: 6px;">
        <span style="color: {color};">●</span>
        <span>Context: {stats.usage_percentage:.0f}%</span>
        <span style="color: #888;">({stats.total_messages} msgs)</span>
        {'<span style="color: #888;">📝 summarized</span>' if stats.has_summary else ''}
    </div>
    """
    
    return indicator_html


def display_memory_stats_sidebar(memory_manager: FrontendMemoryManager, title: str = "Memory Stats"):
    """
    Display memory statistics in the sidebar.
    
    Args:
        memory_manager: The memory manager to display stats for
        title: Title for the stats section
    """
    stats = memory_manager.get_stats()
    
    with st.sidebar.expander(f"🧠 {title}", expanded=False):
        # Progress bar for context usage
        st.progress(min(stats.usage_percentage / 100, 1.0))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", stats.total_messages)
            st.metric("Tokens", stats.total_tokens)
        with col2:
            st.metric("Summary Tokens", stats.summary_tokens)
            st.metric("Usage", f"{stats.usage_percentage:.1f}%")
        
        if stats.has_summary:
            st.success("📝 Conversation summarized")
        
        if stats.needs_summarization:
            st.warning("⚠️ Context near limit")
