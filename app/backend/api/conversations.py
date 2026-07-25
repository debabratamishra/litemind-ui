"""
Conversation CRUD API.

All endpoints require authentication (``get_current_user``) and are scoped to
the calling user via ``user.id``. A user can only read or mutate their own
conversations and messages.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.backend.api.auth_deps import User, get_current_user
from app.backend.conversation_store import (
    ConversationStore,
    get_conversation_store,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


# Dependency wrapper so tests can override the store via dependency_overrides.
def get_store() -> ConversationStore:
    return get_conversation_store()


# ── Request models ─────────────────────────────────────────────────────────────

class CreateConversationRequest(BaseModel):
    title: str = "New Chat"
    conversation_type: str = "chat"


class UpdateConversationRequest(BaseModel):
    title: Optional[str] = None
    summary: Optional[str] = None


class CreateMessageRequest(BaseModel):
    role: str
    content: str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("")
async def list_conversations(
    conversation_type: Optional[str] = None,
    user: User = Depends(get_current_user),
    store: ConversationStore = Depends(get_store),
):
    """List the current user's conversations."""
    conversations = await store.list_conversations(user.id, conversation_type)
    return {"conversations": [c.to_dict() for c in conversations]}


@router.post("")
async def create_conversation(
    body: CreateConversationRequest,
    user: User = Depends(get_current_user),
    store: ConversationStore = Depends(get_store),
):
    """Create a new conversation owned by the current user."""
    try:
        conv = await store.create_conversation(user.id, body.title, body.conversation_type)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return conv.to_dict()


@router.get("/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    user: User = Depends(get_current_user),
    store: ConversationStore = Depends(get_store),
):
    """Get a single conversation (404 if not found or not owned)."""
    conv = await store.get_conversation(conversation_id, user.id)
    if conv is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return conv.to_dict()


@router.patch("/{conversation_id}")
async def update_conversation(
    conversation_id: str,
    body: UpdateConversationRequest,
    user: User = Depends(get_current_user),
    store: ConversationStore = Depends(get_store),
):
    """Rename or set the summary of a conversation (must be owner)."""
    conv = await store.update_conversation(conversation_id, user.id, body.title, body.summary)
    if conv is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return conv.to_dict()


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    user: User = Depends(get_current_user),
    store: ConversationStore = Depends(get_store),
):
    """Delete a conversation owned by the current user."""
    deleted = await store.delete_conversation(conversation_id, user.id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return {"status": "deleted", "id": conversation_id}


@router.get("/{conversation_id}/messages")
async def list_messages(
    conversation_id: str,
    user: User = Depends(get_current_user),
    store: ConversationStore = Depends(get_store),
):
    """List messages for a conversation (must be owner)."""
    messages = await store.get_messages(conversation_id, user.id)
    return {"messages": [m.to_dict() for m in messages]}


@router.post("/{conversation_id}/messages")
async def add_message(
    conversation_id: str,
    body: CreateMessageRequest,
    user: User = Depends(get_current_user),
    store: ConversationStore = Depends(get_store),
):
    """Append a message to a conversation (must be owner)."""
    try:
        msg = await store.add_message(conversation_id, user.id, body.role, body.content)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    if msg is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return msg.to_dict()
