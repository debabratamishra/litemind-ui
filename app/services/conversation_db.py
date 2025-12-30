"""
Conversation Database Service using SQLite.

Provides persistent storage for conversation history with unique identifiers.
Uses SQLite for lightweight, file-based storage that's perfect for local deployments.
"""
import json
import logging
import sqlite3
import uuid
import atexit
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Default database location
DEFAULT_DB_PATH = Path("storage/conversations.db")


def clear_database_file() -> None:
    """Remove the conversation database file if it exists."""
    try:
        DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        if DEFAULT_DB_PATH.exists():
            DEFAULT_DB_PATH.unlink()
            logger.info("Conversation database cleared")
    except Exception as exc:
        logger.warning(f"Failed to clear conversation database: {exc}")


# Clear database on startup
clear_database_file()

# Clear database on interpreter shutdown
atexit.register(clear_database_file)


@dataclass
class ConversationMessage:
    """Represents a single message in a conversation."""
    id: str
    conversation_id: str
    role: str  # "user", "assistant", "system"
    content: str
    created_at: str
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_row(cls, row: Tuple) -> "ConversationMessage":
        return cls(
            id=row[0],
            conversation_id=row[1],
            role=row[2],
            content=row[3],
            created_at=row[4],
            metadata=json.loads(row[5]) if row[5] else None
        )


@dataclass
class Conversation:
    """Represents a conversation with metadata."""
    id: str
    title: str
    conversation_type: str  # "chat" or "rag"
    created_at: str
    updated_at: str
    summary: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    messages: List[ConversationMessage] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "conversation_type": self.conversation_type,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "summary": self.summary,
            "metadata": self.metadata,
            "messages": [m.to_dict() for m in self.messages]
        }
    
    @classmethod
    def from_row(cls, row: Tuple, messages: Optional[List[ConversationMessage]] = None) -> "Conversation":
        return cls(
            id=row[0],
            title=row[1],
            conversation_type=row[2],
            created_at=row[3],
            updated_at=row[4],
            summary=row[5] if len(row) > 5 else None,
            metadata=json.loads(row[6]) if len(row) > 6 and row[6] else None,
            messages=messages or []
        )


def generate_conversation_id() -> str:
    """Generate a unique conversation ID."""
    return f"conv_{uuid.uuid4().hex[:16]}"


def generate_message_id() -> str:
    """Generate a unique message ID."""
    return f"msg_{uuid.uuid4().hex[:12]}"


class ConversationDatabase:
    """
    SQLite-based conversation storage.
    
    Provides CRUD operations for conversations and messages
    with automatic schema management.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the conversation database.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path or DEFAULT_DB_PATH
        
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database schema
        self._init_schema()
        
        logger.info(f"ConversationDatabase initialized at {self.db_path}")
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with automatic cleanup."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _init_schema(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    conversation_type TEXT NOT NULL DEFAULT 'chat',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    summary TEXT,
                    metadata TEXT
                )
            """)
            
            # Messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_id 
                ON messages(conversation_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_type 
                ON conversations(conversation_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_updated 
                ON conversations(updated_at DESC)
            """)
            
            logger.info("Database schema initialized")
    
    # ==================== Conversation Operations ====================
    
    def create_conversation(
        self,
        title: str,
        conversation_type: str = "chat",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Conversation:
        """
        Create a new conversation.
        
        Args:
            title: Conversation title
            conversation_type: Type of conversation ("chat" or "rag")
            metadata: Optional metadata
            
        Returns:
            The created Conversation object
        """
        # Input validation
        if not title or not isinstance(title, str):
            raise ValueError("Title must be a non-empty string")
        if len(title) > 500:
            raise ValueError("Title cannot exceed 500 characters")
        
        # Validate conversation_type
        valid_types = {"chat", "rag"}
        if conversation_type not in valid_types:
            raise ValueError(f"conversation_type must be one of {valid_types}")
        
        conversation_id = generate_conversation_id()
        now = datetime.now().isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO conversations (id, title, conversation_type, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    title,
                    conversation_type,
                    now,
                    now,
                    json.dumps(metadata) if metadata else None
                )
            )
        
        conversation = Conversation(
            id=conversation_id,
            title=title,
            conversation_type=conversation_type,
            created_at=now,
            updated_at=now,
            metadata=metadata
        )
        
        logger.info(f"Created conversation: {conversation_id} - {title}")
        return conversation
    
    def get_conversation(self, conversation_id: str, include_messages: bool = True) -> Optional[Conversation]:
        """
        Get a conversation by ID.
        
        Args:
            conversation_id: The conversation ID
            include_messages: Whether to include messages
            
        Returns:
            Conversation object or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM conversations WHERE id = ?",
                (conversation_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            messages = []
            if include_messages:
                messages = self.get_messages(conversation_id)
            
            return Conversation.from_row(tuple(row), messages)
    
    def list_conversations(
        self,
        conversation_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Conversation]:
        """
        List conversations, optionally filtered by type.
        
        Args:
            conversation_type: Filter by type ("chat" or "rag")
            limit: Maximum number of conversations to return
            offset: Number of conversations to skip
            
        Returns:
            List of Conversation objects (without messages)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if conversation_type:
                cursor.execute(
                    """
                    SELECT * FROM conversations 
                    WHERE conversation_type = ?
                    ORDER BY updated_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (conversation_type, limit, offset)
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM conversations 
                    ORDER BY updated_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset)
                )
            
            rows = cursor.fetchall()
            return [Conversation.from_row(tuple(row)) for row in rows]
    
    def update_conversation(
        self,
        conversation_id: str,
        title: Optional[str] = None,
        summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update a conversation's metadata.
        
        Args:
            conversation_id: The conversation ID
            title: New title (optional)
            summary: New summary (optional)
            metadata: New metadata (optional)
            
        Returns:
            True if updated, False if not found
        """
        updates = []
        params = []
        
        if title is not None:
            updates.append("title = ?")
            params.append(title)
        
        if summary is not None:
            updates.append("summary = ?")
            params.append(summary)
        
        if metadata is not None:
            updates.append("metadata = ?")
            params.append(json.dumps(metadata))
        
        if not updates:
            return True
        
        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(conversation_id)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE conversations SET {', '.join(updates)} WHERE id = ?",
                params
            )
            return cursor.rowcount > 0
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation and all its messages.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            True if deleted, False if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Delete messages first (foreign key)
            cursor.execute(
                "DELETE FROM messages WHERE conversation_id = ?",
                (conversation_id,)
            )
            
            # Delete conversation
            cursor.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,)
            )
            
            deleted = cursor.rowcount > 0
            
            if deleted:
                logger.info(f"Deleted conversation: {conversation_id}")
            
            return deleted
    
    # ==================== Message Operations ====================
    
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationMessage:
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: The conversation ID
            role: Message role ("user", "assistant", "system")
            content: Message content
            metadata: Optional message metadata
            
        Returns:
            The created ConversationMessage object
        """
        # Input validation
        if not conversation_id or not isinstance(conversation_id, str):
            raise ValueError("conversation_id must be a non-empty string")
        
        # Validate role
        valid_roles = {"user", "assistant", "system"}
        if role not in valid_roles:
            raise ValueError(f"role must be one of {valid_roles}")
        
        # Validate content
        if not isinstance(content, str):
            raise ValueError("content must be a string")
        if len(content) > 1000000:  # 1MB limit
            raise ValueError("content cannot exceed 1MB")
        
        message_id = generate_message_id()
        now = datetime.now().isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Insert message
            cursor.execute(
                """
                INSERT INTO messages (id, conversation_id, role, content, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    conversation_id,
                    role,
                    content,
                    now,
                    json.dumps(metadata) if metadata else None
                )
            )
            
            # Update conversation's updated_at timestamp
            cursor.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id)
            )
        
        return ConversationMessage(
            id=message_id,
            conversation_id=conversation_id,
            role=role,
            content=content,
            created_at=now,
            metadata=metadata
        )
    
    def get_messages(
        self,
        conversation_id: str,
        limit: Optional[int] = None
    ) -> List[ConversationMessage]:
        """
        Get all messages for a conversation.
        
        Args:
            conversation_id: The conversation ID
            limit: Maximum number of messages (None for all)
            
        Returns:
            List of ConversationMessage objects
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if limit:
                cursor.execute(
                    """
                    SELECT * FROM messages 
                    WHERE conversation_id = ?
                    ORDER BY created_at ASC
                    LIMIT ?
                    """,
                    (conversation_id, limit)
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM messages 
                    WHERE conversation_id = ?
                    ORDER BY created_at ASC
                    """,
                    (conversation_id,)
                )
            
            rows = cursor.fetchall()
            return [ConversationMessage.from_row(tuple(row)) for row in rows]
    
    def delete_message(self, message_id: str) -> bool:
        """
        Delete a specific message.
        
        Args:
            message_id: The message ID
            
        Returns:
            True if deleted, False if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM messages WHERE id = ?",
                (message_id,)
            )
            return cursor.rowcount > 0
    
    # ==================== Utility Operations ====================
    
    def get_conversation_count(self, conversation_type: Optional[str] = None) -> int:
        """Get the total number of conversations."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if conversation_type:
                cursor.execute(
                    "SELECT COUNT(*) FROM conversations WHERE conversation_type = ?",
                    (conversation_type,)
                )
            else:
                cursor.execute("SELECT COUNT(*) FROM conversations")
            
            return cursor.fetchone()[0]
    
    def search_conversations(
        self,
        query: str,
        conversation_type: Optional[str] = None,
        limit: int = 20
    ) -> List[Conversation]:
        """
        Search conversations by title or message content.
        
        Args:
            query: Search query
            conversation_type: Filter by type
            limit: Maximum results
            
        Returns:
            List of matching Conversation objects
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            def _escape_like(raw: str) -> str:
                """Escape SQLite LIKE wildcards in user input."""
                return (
                    raw.replace("\\", "\\\\")
                    .replace("%", "\\%")
                    .replace("_", "\\_")
                    .replace("[", "\\[")
                )

            escaped_query = _escape_like(query)
            search_pattern = f"%{escaped_query}%"
            
            if conversation_type:
                cursor.execute(
                    """
                    SELECT DISTINCT c.* FROM conversations c
                    LEFT JOIN messages m ON c.id = m.conversation_id
                    WHERE c.conversation_type = ?
                    AND (c.title LIKE ? ESCAPE '\\' OR m.content LIKE ? ESCAPE '\\')
                    ORDER BY c.updated_at DESC
                    LIMIT ?
                    """,
                    (conversation_type, search_pattern, search_pattern, limit)
                )
            else:
                cursor.execute(
                    """
                    SELECT DISTINCT c.* FROM conversations c
                    LEFT JOIN messages m ON c.id = m.conversation_id
                    WHERE c.title LIKE ? ESCAPE '\\' OR m.content LIKE ? ESCAPE '\\'
                    ORDER BY c.updated_at DESC
                    LIMIT ?
                    """,
                    (search_pattern, search_pattern, limit)
                )
            
            rows = cursor.fetchall()
            return [Conversation.from_row(tuple(row)) for row in rows]
    
    def generate_title_from_message(self, message: str, max_words: int = 5) -> str:
        """
        Generate a brief conversation title from the first message.
        
        Args:
            message: The first message content
            max_words: Maximum number of words in title (default 5)
            
        Returns:
            Generated title string (brief, max 5 words)
        """
        # Clean the message
        title = message.strip()
        
        # Remove newlines and extra spaces
        title = ' '.join(title.replace('\n', ' ').replace('\r', '').split())
        
        # Split into words and take first max_words
        words = title.split()
        if len(words) > max_words:
            title = ' '.join(words[:max_words]) + "..."
        
        # Ensure reasonable length even if words are very long
        if len(title) > 40:
            title = title[:37] + "..."
        
        return title if title else "New Chat"


# Singleton instance
_db_instance: Optional[ConversationDatabase] = None


def get_conversation_db() -> ConversationDatabase:
    """Get the singleton ConversationDatabase instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = ConversationDatabase()
    return _db_instance
