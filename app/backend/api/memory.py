"""User-level persistent memory CRUD endpoints (all require authentication)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.backend.api.auth_deps import User, get_current_user
from app.backend.models.api_models import MemoryContentRequest, MemoryRecordResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/memory", tags=["memory"])


def _get_user_memory_store():
    from app.backend.user_memory_store import get_user_memory_store
    return get_user_memory_store()


def _response(record) -> MemoryRecordResponse:
    return MemoryRecordResponse(
        id=record.id,
        content=record.content,
        source=record.source,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


@router.get("", response_model=list[MemoryRecordResponse])
async def list_memories(user: User = Depends(get_current_user)):
    records = await _get_user_memory_store().list_memories(user.id)
    return [_response(r) for r in records]


@router.post("", response_model=MemoryRecordResponse)
async def add_memory(request: MemoryContentRequest, user: User = Depends(get_current_user)):
    content = request.content.strip()
    if not content:
        raise HTTPException(status_code=422, detail="Memory content cannot be empty")
    store = _get_user_memory_store()
    record = await store.add_memory(user.id, content, source="manual")
    return _response(record)


@router.put("/{memory_id}")
async def update_memory(memory_id: str, request: MemoryContentRequest, user: User = Depends(get_current_user)):
    content = request.content.strip()
    if not content:
        raise HTTPException(status_code=422, detail="Memory content cannot be empty")
    updated = await _get_user_memory_store().update_memory(user.id, memory_id, content)
    if not updated:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"status": "updated", "id": memory_id}


@router.delete("/{memory_id}")
async def delete_memory(memory_id: str, user: User = Depends(get_current_user)):
    deleted = await _get_user_memory_store().delete_memory(user.id, memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"status": "deleted", "id": memory_id}


@router.post("/clear")
async def clear_memories(user: User = Depends(get_current_user)):
    count = await _get_user_memory_store().clear_memories(user.id)
    return {"status": "cleared", "deleted": count}
