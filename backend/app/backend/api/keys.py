"""Encrypted API-key management endpoints (vault-backed; metadata only)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.app.backend.api.deps import User, get_system_user
from backend.app.kernel.keyvault import get_key_vault
from backend.app.kernel.router import CAPABILITIES

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/keys", tags=["keys"])


class KeyUpsertRequest(BaseModel):
    kind: str
    provider: str
    label: str = ""
    value: str


@router.get("")
async def list_keys(user: User = Depends(get_system_user)) -> list[dict]:
    """List stored credentials: metadata + configured flag, never the token."""
    rows = await get_key_vault().list_credentials()
    return [{**row, "configured": True} for row in rows]


@router.put("")
async def put_key(request: KeyUpsertRequest, user: User = Depends(get_system_user)) -> dict:
    """Encrypt and store a key; same (kind, provider) upserts in place."""
    kind = request.kind.strip()
    if kind not in CAPABILITIES:
        raise HTTPException(
            status_code=422, detail=f"kind must be one of: {', '.join(CAPABILITIES)}"
        )
    provider = request.provider.strip()
    if not provider:
        raise HTTPException(status_code=422, detail="provider cannot be empty")
    if not request.value.strip():
        raise HTTPException(status_code=422, detail="value cannot be empty")

    credential_id = await get_key_vault().set_credential(kind, provider, request.label, request.value)
    return {
        "id": credential_id,
        "kind": kind,
        "provider": provider,
        "label": request.label,
        "configured": True,
    }


@router.delete("/{credential_id}")
async def delete_key(credential_id: str, user: User = Depends(get_system_user)) -> dict:
    try:
        deleted = await get_key_vault().delete_credential(credential_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Credential not found") from None
    if not deleted:
        raise HTTPException(status_code=404, detail="Credential not found")
    return {"status": "deleted", "id": credential_id}
