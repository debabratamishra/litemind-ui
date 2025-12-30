"""
Conversation History Sidebar Component.

Provides an aesthetically pleasing sidebar for managing conversation history,
including creating new conversations, switching between existing ones,
and deleting old conversations.
"""
import logging
from datetime import datetime
from typing import Optional

import streamlit as st

from ...services.conversation_db import (
    get_conversation_db,
    Conversation
)

logger = logging.getLogger(__name__)


# Custom CSS for the conversation sidebar
SIDEBAR_CSS = """
<style>
/* Conversation history container */
.conversation-history {
    max-height: 400px;
    overflow-y: auto;
    padding-right: 4px;
}

/* Individual conversation item */
.conversation-item {
    display: flex;
    align-items: center;
    padding: 10px 12px;
    margin: 4px 0;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s ease;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid transparent;
}

.conversation-item:hover {
    background: rgba(255, 255, 255, 0.08);
    border-color: rgba(255, 255, 255, 0.1);
}

.conversation-item.active {
    background: rgba(99, 102, 241, 0.15);
    border-color: rgba(99, 102, 241, 0.3);
}

/* Conversation icon */
.conversation-icon {
    width: 32px;
    height: 32px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-right: 10px;
    font-size: 14px;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    flex-shrink: 0;
}

.conversation-icon.rag {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
}

/* Conversation details */
.conversation-details {
    flex: 1;
    min-width: 0;
    overflow: hidden;
}

.conversation-title {
    font-size: 0.9em;
    font-weight: 500;
    color: #e5e7eb;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: 2px;
}

.conversation-meta {
    font-size: 0.75em;
    color: #9ca3af;
}

/* New conversation button */
.new-conversation-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 12px;
    margin: 8px 0;
    border-radius: 8px;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
    border: none;
}

.new-conversation-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
}

/* Empty state */
.empty-conversations {
    text-align: center;
    padding: 24px 16px;
    color: #9ca3af;
    font-size: 0.9em;
}

.empty-conversations-icon {
    font-size: 2em;
    margin-bottom: 8px;
    opacity: 0.5;
}

/* Delete button */
.delete-btn {
    opacity: 0;
    transition: opacity 0.2s ease;
    padding: 4px;
    border-radius: 4px;
    color: #ef4444;
    flex-shrink: 0;
}

.conversation-item:hover .delete-btn {
    opacity: 1;
}

/* Scrollbar styling */
.conversation-history::-webkit-scrollbar {
    width: 6px;
}

.conversation-history::-webkit-scrollbar-track {
    background: transparent;
}

.conversation-history::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.2);
    border-radius: 3px;
}

.conversation-history::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.3);
}

/* Section header */
.section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 4px;
    margin-bottom: 8px;
}

.section-title {
    font-size: 0.85em;
    font-weight: 600;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
</style>
"""


def format_time_ago(timestamp_str: str) -> str:
    """Format a timestamp as a relative time string."""
    try:
        timestamp = datetime.fromisoformat(timestamp_str)
        now = datetime.now()
        diff = now - timestamp
        
        seconds = diff.total_seconds()
        
        if seconds < 60:
            return "Just now"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes}m ago"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours}h ago"
        elif seconds < 604800:
            days = int(seconds / 86400)
            return f"{days}d ago"
        else:
            return timestamp.strftime("%b %d")
    except Exception:
        return ""


class ConversationHistorySidebar:
    """
    Sidebar component for managing conversation history.
    
    Features:
    - Display list of past conversations
    - Create new conversations
    - Switch between conversations
    - Delete conversations
    - Search conversations
    """
    
    def __init__(self, conversation_type: str = "chat"):
        """
        Initialize the conversation history sidebar.
        
        Args:
            conversation_type: Type of conversations to display ("chat" or "rag")
        """
        self.conversation_type = conversation_type
        self.db = get_conversation_db()
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state for conversation management."""
        # Current active conversation ID
        key_prefix = self.conversation_type
        
        if f"{key_prefix}_active_conversation_id" not in st.session_state:
            st.session_state[f"{key_prefix}_active_conversation_id"] = None
        
        if f"{key_prefix}_show_delete_confirm" not in st.session_state:
            st.session_state[f"{key_prefix}_show_delete_confirm"] = None
    
    @property
    def active_conversation_id(self) -> Optional[str]:
        """Get the currently active conversation ID."""
        return st.session_state.get(f"{self.conversation_type}_active_conversation_id")
    
    @active_conversation_id.setter
    def active_conversation_id(self, value: Optional[str]):
        """Set the currently active conversation ID."""
        st.session_state[f"{self.conversation_type}_active_conversation_id"] = value
    
    def create_new_conversation(self, first_message: Optional[str] = None) -> Conversation:
        """
        Create a new conversation and set it as active.
        
        Args:
            first_message: Optional first message to generate title from
            
        Returns:
            The newly created Conversation
        """
        title = "New Conversation"
        if first_message:
            title = self.db.generate_title_from_message(first_message)
        
        conversation = self.db.create_conversation(
            title=title,
            conversation_type=self.conversation_type
        )
        
        self.active_conversation_id = conversation.id
        
        # Clear current messages in session state
        messages_key = "chat_messages" if self.conversation_type == "chat" else "rag_messages"
        st.session_state[messages_key] = []
        
        # Clear summary
        summary_key = f"{self.conversation_type}_conversation_summary"
        st.session_state[summary_key] = None
        
        logger.info(f"Created new {self.conversation_type} conversation: {conversation.id}")
        return conversation
    
    def load_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """
        Load a conversation and its messages into session state.
        
        Args:
            conversation_id: The conversation ID to load
            
        Returns:
            The loaded Conversation or None if not found
        """
        conversation = self.db.get_conversation(conversation_id, include_messages=True)
        
        if not conversation:
            logger.warning(f"Conversation not found: {conversation_id}")
            return None
        
        # Set as active
        self.active_conversation_id = conversation_id
        
        # Load messages into session state
        messages_key = "chat_messages" if self.conversation_type == "chat" else "rag_messages"
        st.session_state[messages_key] = [
            {
                "role": msg.role,
                "content": msg.content,
                **(msg.metadata or {})
            }
            for msg in conversation.messages
        ]
        
        # Load summary if available
        summary_key = f"{self.conversation_type}_conversation_summary"
        st.session_state[summary_key] = conversation.summary
        
        logger.info(f"Loaded conversation: {conversation_id} with {len(conversation.messages)} messages")
        return conversation
    
    def save_message(self, role: str, content: str, metadata: Optional[dict] = None):
        """
        Save a message to the current active conversation.
        
        Args:
            role: Message role
            content: Message content
            metadata: Optional metadata
        """
        if not self.active_conversation_id:
            # Create a new conversation with this as the first message
            conversation = self.create_new_conversation(
                first_message=content if role == "user" else None
            )
            self.active_conversation_id = conversation.id
        
        # Add message to database
        self.db.add_message(
            conversation_id=self.active_conversation_id,
            role=role,
            content=content,
            metadata=metadata
        )
        
        # Update title if this is the first user message
        conversation = self.db.get_conversation(self.active_conversation_id, include_messages=False)
        if conversation and conversation.title == "New Conversation" and role == "user":
            new_title = self.db.generate_title_from_message(content)
            self.db.update_conversation(self.active_conversation_id, title=new_title)
    
    def update_conversation_summary(self, summary: str):
        """Update the summary for the current conversation."""
        if self.active_conversation_id:
            self.db.update_conversation(
                self.active_conversation_id,
                summary=summary
            )
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation.
        
        Args:
            conversation_id: The conversation ID to delete
            
        Returns:
            True if deleted successfully
        """
        success = self.db.delete_conversation(conversation_id)
        
        if success and conversation_id == self.active_conversation_id:
            # Clear active conversation
            self.active_conversation_id = None
            
            # Clear session state messages
            messages_key = "chat_messages" if self.conversation_type == "chat" else "rag_messages"
            st.session_state[messages_key] = []
        
        return success
    
    def render(self):
        """Render the conversation history sidebar."""
        # Inject custom CSS
        st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)
        
        # Get conversations
        conversations = self.db.list_conversations(
            conversation_type=self.conversation_type,
            limit=50
        )
        
        # Check if there's an active conversation with messages
        messages_key = "chat_messages" if self.conversation_type == "chat" else "rag_messages"
        has_active_messages = len(st.session_state.get(messages_key, [])) > 0
        
        # Header row with new conversation button
        st.sidebar.markdown("---")
        col_title, col_btn = st.sidebar.columns([4, 1])
        with col_title:
            st.markdown("#### Conversation History")
        with col_btn:
            # Only allow new conversation if user has messages in current session
            if has_active_messages:
                if st.button("➕", key=f"new_{self.conversation_type}_conv", help="New conversation"):
                    self._start_new_conversation()
                    st.rerun()
        
        
        if not conversations:
            st.sidebar.caption("Start chatting to build conversation history")
            return
        
        # Render conversation list
        for conv in conversations:
            self._render_conversation_item(conv)
    
    def _start_new_conversation(self):
        """Start a fresh conversation without creating a DB entry yet."""
        # Just clear the current session - DB entry will be created on first message
        self.active_conversation_id = None
        messages_key = "chat_messages" if self.conversation_type == "chat" else "rag_messages"
        st.session_state[messages_key] = []
        summary_key = f"{self.conversation_type}_conversation_summary"
        st.session_state[summary_key] = None
        logger.info(f"Started new {self.conversation_type} conversation session")
    
    def _render_conversation_item(self, conversation: Conversation):
        """Render a single conversation item in the sidebar."""
        is_active = conversation.id == self.active_conversation_id
        show_delete = st.session_state.get(f"{self.conversation_type}_show_delete_confirm") == conversation.id
        
        # Show delete confirmation if this conversation is selected for deletion
        if show_delete:
            st.sidebar.markdown(f"Delete **{conversation.title[:20]}{'...' if len(conversation.title) > 20 else ''}**?")
            col_yes, col_no = st.sidebar.columns(2)
            with col_yes:
                if st.button("Yes", key=f"confirm_del_{conversation.id}", type="primary", use_container_width=True):
                    self.delete_conversation(conversation.id)
                    st.session_state[f"{self.conversation_type}_show_delete_confirm"] = None
                    st.rerun()
            with col_no:
                if st.button("No", key=f"cancel_del_{conversation.id}", use_container_width=True):
                    st.session_state[f"{self.conversation_type}_show_delete_confirm"] = None
                    st.rerun()
            return
        
        # Simple clean button with just the title
        button_label = conversation.title
        if len(button_label) > 25:
            button_label = button_label[:22] + "..."
        
        # Use columns for conversation button and delete
        col1, col2 = st.sidebar.columns([5, 1])
        
        with col1:
            button_type = "primary" if is_active else "secondary"
            if st.button(
                button_label,
                key=f"conv_{conversation.id}",
                type=button_type,
                use_container_width=True
            ):
                if not is_active:
                    self.load_conversation(conversation.id)
                    st.rerun()
        
        with col2:
            if st.button("×", key=f"del_{conversation.id}", help="Delete"):
                st.session_state[f"{self.conversation_type}_show_delete_confirm"] = conversation.id
                st.rerun()
    
    def render_compact(self):
        """Render a compact version of the conversation sidebar."""
        # Inject custom CSS
        st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)
        
        # Check if there's an active conversation with messages
        messages_key = "chat_messages" if self.conversation_type == "chat" else "rag_messages"
        has_active_messages = len(st.session_state.get(messages_key, [])) > 0
        
        # Header row
        col1, col2 = st.sidebar.columns([3, 1])
        
        with col1:
            st.sidebar.markdown(
                f"**{'💬' if self.conversation_type == 'chat' else '📚'} History**"
            )
        
        with col2:
            # Only show new button if there are active messages
            if has_active_messages:
                if st.button("➕", key=f"new_{self.conversation_type}_compact", help="New conversation"):
                    self._start_new_conversation()
                    st.rerun()
        
        # Get recent conversations
        conversations = self.db.list_conversations(
            conversation_type=self.conversation_type,
            limit=10
        )
        
        if conversations:
            # Create a selectbox for conversation selection
            conv_options = {
                conv.title[:25] + ('...' if len(conv.title) > 25 else ''): conv.id
                for conv in conversations
            }
            
            # Find current selection
            current_label = None
            if self.active_conversation_id:
                for label, cid in conv_options.items():
                    if cid == self.active_conversation_id:
                        current_label = label
                        break
            
            if current_label and current_label in conv_options:
                current_index = list(conv_options.keys()).index(current_label)
            else:
                current_index = 0
            
            selected = st.sidebar.selectbox(
                "Conversation:",
                options=list(conv_options.keys()),
                index=current_index,
                key=f"{self.conversation_type}_conv_select",
                label_visibility="collapsed"
            )
            
            selected_id = conv_options.get(selected)
            
            if selected_id and selected_id != self.active_conversation_id:
                self.load_conversation(selected_id)
                st.rerun()


# Convenience functions for use in pages
def get_chat_sidebar() -> ConversationHistorySidebar:
    """Get the chat conversation sidebar instance."""
    if "chat_conversation_sidebar" not in st.session_state:
        st.session_state.chat_conversation_sidebar = ConversationHistorySidebar("chat")
    return st.session_state.chat_conversation_sidebar


def get_rag_sidebar() -> ConversationHistorySidebar:
    """Get the RAG conversation sidebar instance."""
    if "rag_conversation_sidebar" not in st.session_state:
        st.session_state.rag_conversation_sidebar = ConversationHistorySidebar("rag")
    return st.session_state.rag_conversation_sidebar
