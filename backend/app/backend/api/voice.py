"""Realtime voice session endpoints: ElevenLabs and Sarvam Conversational AI."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.app.backend.api.deps import User, get_system_user
from backend.app.services.voice_pipeline import (
    VoiceConfigurationError,
    get_convai_signed_url,
    resolve_elevenlabs_credentials,
)

logger = logging.getLogger(__name__)

router = APIRouter()

pcs_map: dict[str, Any] = {}


class VoiceSessionRequest(BaseModel):
    agent_id: Optional[str] = None
    api_key: Optional[str] = None


class SarvamSessionRequest(BaseModel):
    api_key: Optional[str] = None
    org_id: Optional[str] = None
    workspace_id: Optional[str] = None
    app_id: Optional[str] = None


async def _resolve_sarvam_credentials(
    explicit_key: Optional[str] = None,
    explicit_org_id: Optional[str] = None,
    explicit_workspace_id: Optional[str] = None,
    explicit_app_id: Optional[str] = None,
) -> tuple[str, str, str, str]:
    """Resolve Sarvam credentials: explicit param > KeyVault > env var."""

    def _clean(v: Optional[str]) -> Optional[str]:
        return v.strip() if v and v.strip() else None

    api_key = _clean(explicit_key)
    org_id = _clean(explicit_org_id)
    workspace_id = _clean(explicit_workspace_id)
    app_id = _clean(explicit_app_id)

    if not all([api_key, org_id, workspace_id, app_id]):
        try:
            from backend.app.kernel.keyvault import get_key_vault

            vault = get_key_vault()
            if not api_key:
                val = await vault.get_credential('voice', 'sarvam')
                if val:
                    api_key = val
            if not org_id:
                val = await vault.get_credential('voice', 'sarvam_org_id')
                if val:
                    org_id = val
            if not workspace_id:
                val = await vault.get_credential('voice', 'sarvam_workspace_id')
                if val:
                    workspace_id = val
            if not app_id:
                val = await vault.get_credential('voice', 'sarvam_app_id')
                if val:
                    app_id = val
        except Exception:
            logger.warning('KeyVault lookup for Sarvam credentials failed', exc_info=True)

    api_key = api_key or os.getenv('SARVAM_API_KEY')
    org_id = org_id or os.getenv('SARVAM_ORG_ID')
    workspace_id = workspace_id or os.getenv('SARVAM_WORKSPACE_ID')
    app_id = app_id or os.getenv('SARVAM_APP_ID')

    missing = [name for name, val in [
        ('API key (SARVAM_API_KEY)', api_key),
        ('Org ID (SARVAM_ORG_ID)', org_id),
        ('Workspace ID (SARVAM_WORKSPACE_ID)', workspace_id),
        ('App ID (SARVAM_APP_ID)', app_id),
    ] if not val]

    if missing:
        raise VoiceConfigurationError(
            f'Sarvam voice mode is missing: {", ".join(missing)}. '
            'Save them in Settings → Providers or set the environment variables.'
        )

    # All four are guaranteed non-None by the missing-values check above.
    assert api_key and org_id and workspace_id and app_id
    return api_key, org_id, workspace_id, app_id


@router.post('/api/voice/session')
async def create_voice_session(
    request: Optional[VoiceSessionRequest] = None,
    user: User = Depends(get_system_user),
) -> dict:
    """Create a direct ElevenLabs Conversational AI session.

    Resolves credentials (explicit param > key vault > env var) and requests a
    secure signed WebSocket URL from ElevenLabs.
    """
    req_agent_id = request.agent_id if request else None
    req_api_key = request.api_key if request else None

    try:
        api_key, agent_id = await resolve_elevenlabs_credentials(
            explicit_key=req_api_key, explicit_agent_id=req_agent_id
        )
        signed_url = await get_convai_signed_url(agent_id=agent_id, api_key=api_key)
        return {'signed_url': signed_url, 'agent_id': agent_id}
    except VoiceConfigurationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception('Failed to initialize ElevenLabs voice session')
        raise HTTPException(status_code=500, detail='Failed to initialize voice session.') from exc


@router.get('/api/voice/session')
async def get_voice_session(
    agent_id: Optional[str] = None,
    user: User = Depends(get_system_user),
) -> dict:
    """GET variant of create_voice_session."""
    return await create_voice_session(VoiceSessionRequest(agent_id=agent_id), user=user)


@router.post('/api/voice/sarvam-session')
async def create_sarvam_session(
    request: Optional[SarvamSessionRequest] = None,
    user: User = Depends(get_system_user),
) -> dict:
    """Return resolved Sarvam Conversations credentials to the client SDK.

    The API key is never stored in the frontend; this endpoint resolves it
    from the vault or environment and returns it only for the duration of the
    session request.  The browser SDK uses it directly to authenticate the
    WebSocket connection.
    """
    r = request or SarvamSessionRequest()
    try:
        api_key, org_id, workspace_id, app_id = await _resolve_sarvam_credentials(
            explicit_key=r.api_key,
            explicit_org_id=r.org_id,
            explicit_workspace_id=r.workspace_id,
            explicit_app_id=r.app_id,
        )
        return {
            'api_key': api_key,
            'org_id': org_id,
            'workspace_id': workspace_id,
            'app_id': app_id,
        }
    except VoiceConfigurationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception('Failed to resolve Sarvam voice session credentials')
        raise HTTPException(status_code=500, detail='Failed to initialize Sarvam voice session.') from exc


@router.post('/api/voice/offer')
async def offer(request: dict, user: User = Depends(get_system_user)) -> dict:
    """Backwards-compatible offer endpoint; guides clients to /api/voice/session."""
    pc_id = request.get('pc_id')
    if not pc_id:
        return {'error': 'pc_id required'}
    pcs_map[pc_id] = True
    return {
        'pc_id': pc_id,
        'type': 'answer',
        'sdp': 'v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n',
        'notice': 'WebRTC voice pipeline upgraded to ElevenLabs Conversational AI. Use /api/voice/session.',
        'error': 'WebRTC voice pipeline upgraded to ElevenLabs Conversational AI. Use /api/voice/session.',
    }
