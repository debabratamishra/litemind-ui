"""
Conversation Memory Service for multi-turn conversations with summarization.

This service handles:
- Storing conversation history
- Token counting to prevent context explosion
- Automatic summarization when context grows too large
- Memory pruning strategies
"""
import os
import json
import logging
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import hashlib

from .ollama import stream_ollama

logger = logging.getLogger(__name__)

# Default token limits and thresholds
DEFAULT_MAX_CONTEXT_TOKENS = 24000  # Leave ~8K for response in 32K context
DEFAULT_SUMMARIZE_THRESHOLD = 0.75  # Summarize when at 75% of max tokens
DEFAULT_SUMMARY_MAX_TOKENS = 2000  # Maximum tokens for summary

# Approximate tokens per character (conservative estimate)
CHARS_PER_TOKEN = 4


@dataclass
class Message:
    """Represents a single message in a conversation."""
    role: str  # "user", "assistant", or "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    token_count: int = 0
    is_summary: bool = False  # True if this message is a summary of previous messages
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "token_count": self.token_count,
            "is_summary": self.is_summary
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now(),
            token_count=data.get("token_count", 0),
            is_summary=data.get("is_summary", False)
        )


@dataclass
class ConversationContext:
    """Holds the full context for a conversation session."""
    session_id: str
    messages: List[Message] = field(default_factory=list)
    total_tokens: int = 0
    summary: Optional[str] = None  # Summary of pruned messages
    summary_tokens: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "messages": [m.to_dict() for m in self.messages],
            "total_tokens": self.total_tokens,
            "summary": self.summary,
            "summary_tokens": self.summary_tokens,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat()
        }


class ConversationMemoryService:
    """
    Manages conversation memory with automatic summarization to prevent context explosion.
    
    Key features:
    - Token counting for all messages
    - Automatic summarization when approaching token limit
    - Sliding window with summary for efficient context management
    - Session-based memory isolation
    """
    
    def __init__(
        self,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        summarize_threshold: float = DEFAULT_SUMMARIZE_THRESHOLD,
        summary_max_tokens: int = DEFAULT_SUMMARY_MAX_TOKENS,
        model_for_summary: str = "gemma3:1b"
    ):
        """
        Initialize the conversation memory service.
        
        Args:
            max_context_tokens: Maximum tokens allowed in context
            summarize_threshold: Fraction of max tokens to trigger summarization
            summary_max_tokens: Maximum tokens for the summary
            model_for_summary: Model to use for generating summaries
        """
        self.max_context_tokens = max_context_tokens
        self.summarize_threshold = summarize_threshold
        self.summary_max_tokens = summary_max_tokens
        self.model_for_summary = model_for_summary
        
        # In-memory storage for active sessions
        self._sessions: Dict[str, ConversationContext] = {}
        
        logger.info(
            f"ConversationMemoryService initialized: "
            f"max_tokens={max_context_tokens}, "
            f"threshold={summarize_threshold}, "
            f"summary_max={summary_max_tokens}"
        )
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.
        Uses a conservative character-based estimation.
        
        For more accurate counting, you could integrate tiktoken or similar,
        but this provides a good approximation.
        """
        if not text:
            return 0
        # Conservative estimate: ~4 characters per token
        return max(1, len(text) // CHARS_PER_TOKEN)
    
    def get_or_create_session(self, session_id: str) -> ConversationContext:
        """Get existing session or create a new one."""
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationContext(session_id=session_id)
            logger.info(f"Created new conversation session: {session_id}")
        return self._sessions[session_id]
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        is_summary: bool = False
    ) -> Message:
        """
        Add a message to the conversation history.
        
        Args:
            session_id: The session identifier
            role: Message role (user, assistant, system)
            content: Message content
            is_summary: Whether this is a summary message
            
        Returns:
            The created Message object
        """
        session = self.get_or_create_session(session_id)
        
        token_count = self.estimate_tokens(content)
        message = Message(
            role=role,
            content=content,
            token_count=token_count,
            is_summary=is_summary
        )
        
        session.messages.append(message)
        session.total_tokens += token_count
        session.last_activity = datetime.now()
        
        logger.debug(
            f"Added message to session {session_id}: "
            f"role={role}, tokens={token_count}, total={session.total_tokens}"
        )
        
        return message
    
    def get_messages_for_context(
        self,
        session_id: str,
        include_summary: bool = True
    ) -> List[Dict[str, str]]:
        """
        Get messages formatted for LLM context.
        
        Returns a list of messages suitable for sending to the LLM,
        including any conversation summary if available.
        
        Args:
            session_id: The session identifier
            include_summary: Whether to include the summary in context
            
        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        session = self.get_or_create_session(session_id)
        messages = []
        
        # Add summary as system context if available
        if include_summary and session.summary:
            messages.append({
                "role": "system",
                "content": f"Summary of previous conversation:\n{session.summary}"
            })
        
        # Add current messages
        for msg in session.messages:
            if not msg.is_summary:  # Skip summary markers
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        return messages
    
    def get_context_token_count(self, session_id: str) -> int:
        """Get the current token count for a session."""
        session = self.get_or_create_session(session_id)
        return session.total_tokens + session.summary_tokens
    
    def needs_summarization(self, session_id: str) -> bool:
        """Check if the session needs summarization based on token count."""
        session = self.get_or_create_session(session_id)
        threshold_tokens = int(self.max_context_tokens * self.summarize_threshold)
        current_tokens = session.total_tokens + session.summary_tokens
        return current_tokens >= threshold_tokens
    
    async def summarize_if_needed(
        self,
        session_id: str,
        force: bool = False
    ) -> bool:
        """
        Summarize the conversation history if needed.
        
        This method:
        1. Checks if summarization is needed (or forced)
        2. Generates a summary of older messages
        3. Removes the summarized messages
        4. Stores the summary for future context
        
        Args:
            session_id: The session identifier
            force: Force summarization regardless of token count
            
        Returns:
            True if summarization was performed, False otherwise
        """
        if not force and not self.needs_summarization(session_id):
            return False
        
        session = self.get_or_create_session(session_id)
        
        if len(session.messages) < 4:
            # Not enough messages to summarize
            logger.debug(f"Session {session_id}: Too few messages to summarize")
            return False
        
        # Keep the most recent messages (last 2-4 turns)
        keep_recent = 4  # Keep last 4 messages
        messages_to_summarize = session.messages[:-keep_recent]
        messages_to_keep = session.messages[-keep_recent:]
        
        if not messages_to_summarize:
            return False
        
        logger.info(
            f"Summarizing session {session_id}: "
            f"{len(messages_to_summarize)} messages to summarize, "
            f"{len(messages_to_keep)} to keep"
        )
        
        # Generate summary
        summary = await self._generate_summary(
            messages_to_summarize,
            session.summary  # Include existing summary for continuity
        )
        
        if not summary:
            logger.warning(f"Failed to generate summary for session {session_id}")
            return False
        
        # Update session
        tokens_removed = sum(m.token_count for m in messages_to_summarize)
        session.summary = summary
        session.summary_tokens = self.estimate_tokens(summary)
        session.messages = messages_to_keep
        session.total_tokens = sum(m.token_count for m in messages_to_keep)
        
        logger.info(
            f"Summarization complete for session {session_id}: "
            f"removed {tokens_removed} tokens, "
            f"summary has {session.summary_tokens} tokens, "
            f"new total: {session.total_tokens + session.summary_tokens}"
        )
        
        return True
    
    async def _generate_summary(
        self,
        messages: List[Message],
        existing_summary: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate a summary of the given messages.
        
        Args:
            messages: List of messages to summarize
            existing_summary: Existing summary to incorporate
            
        Returns:
            Generated summary text, or None if failed
        """
        # Format messages for summarization
        conversation_text = []
        if existing_summary:
            conversation_text.append(f"Previous context: {existing_summary}")
        
        for msg in messages:
            conversation_text.append(f"{msg.role.capitalize()}: {msg.content}")
        
        full_text = "\n\n".join(conversation_text)
        
        # Truncate if too long (roughly limit to 8K chars for input)
        max_chars = 8000
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars] + "\n...[truncated for summarization]"
        
        summary_prompt = f"""Summarize the following conversation concisely, capturing:
1. Key topics discussed
2. Important decisions or conclusions
3. Any specific requests or preferences expressed
4. Essential context needed for future responses

Keep the summary under 500 words. Focus on information that would be relevant for continuing the conversation.

Conversation:
{full_text}

Summary:"""

        try:
            summary_parts = []
            messages_for_llm = [
                {"role": "system", "content": "You are a helpful assistant that creates concise, informative summaries of conversations."},
                {"role": "user", "content": summary_prompt}
            ]
            
            async for chunk in stream_ollama(
                messages_for_llm,
                model=self.model_for_summary,
                temperature=0.3  # Lower temperature for more consistent summaries
            ):
                summary_parts.append(chunk)
            
            summary = "".join(summary_parts).strip()
            
            if summary:
                logger.debug(f"Generated summary with {len(summary)} characters")
                return summary
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return None
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear all messages and summary for a session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            True if session was cleared, False if it didn't exist
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Cleared session: {session_id}")
            return True
        return False
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics about a session."""
        session = self.get_or_create_session(session_id)
        return {
            "session_id": session_id,
            "message_count": len(session.messages),
            "total_tokens": session.total_tokens,
            "summary_tokens": session.summary_tokens,
            "has_summary": session.summary is not None,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat()
        }
    
    def prepare_context_for_llm(
        self,
        session_id: str,
        new_message: str,
        system_prompt: Optional[str] = None
    ) -> Tuple[List[Dict[str, str]], int]:
        """
        Prepare the full context for sending to the LLM.
        
        This method:
        1. Gets existing conversation context
        2. Adds the new user message
        3. Calculates total tokens
        4. Returns formatted messages ready for LLM
        
        Args:
            session_id: The session identifier
            new_message: The new user message to add
            system_prompt: Optional system prompt to include
            
        Returns:
            Tuple of (formatted messages list, total token count)
        """
        session = self.get_or_create_session(session_id)
        messages = []
        total_tokens = 0
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            total_tokens += self.estimate_tokens(system_prompt)
        
        # Add summary if available
        if session.summary:
            summary_context = f"Summary of previous conversation:\n{session.summary}"
            messages.append({"role": "system", "content": summary_context})
            total_tokens += session.summary_tokens
        
        # Add conversation history
        for msg in session.messages:
            if not msg.is_summary:
                messages.append({"role": msg.role, "content": msg.content})
                total_tokens += msg.token_count
        
        # Add new message
        messages.append({"role": "user", "content": new_message})
        total_tokens += self.estimate_tokens(new_message)
        
        return messages, total_tokens


# Singleton instance for global access
_memory_service: Optional[ConversationMemoryService] = None


def get_memory_service() -> ConversationMemoryService:
    """Get or create the global conversation memory service instance."""
    global _memory_service
    if _memory_service is None:
        # Load configuration from environment
        max_tokens = int(os.getenv("MAX_CONTEXT_TOKENS", DEFAULT_MAX_CONTEXT_TOKENS))
        threshold = float(os.getenv("SUMMARIZE_THRESHOLD", DEFAULT_SUMMARIZE_THRESHOLD))
        summary_max = int(os.getenv("SUMMARY_MAX_TOKENS", DEFAULT_SUMMARY_MAX_TOKENS))
        model = os.getenv("SUMMARY_MODEL", "gemma3:1b")
        
        _memory_service = ConversationMemoryService(
            max_context_tokens=max_tokens,
            summarize_threshold=threshold,
            summary_max_tokens=summary_max,
            model_for_summary=model
        )
    return _memory_service


def generate_session_id(prefix: str = "chat") -> str:
    """Generate a unique session ID."""
    timestamp = datetime.now().isoformat()
    unique_part = hashlib.md5(f"{timestamp}{os.urandom(8).hex()}".encode()).hexdigest()[:12]
    return f"{prefix}_{unique_part}"
