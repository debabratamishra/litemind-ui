"""Realtime voice session management: ElevenLabs Conversational AI.

Provides direct audio-in/audio-out conversational speech sessions.
The backend resolves the user's ElevenLabs API key and Agent ID from the
KeyVault or environment variables, requests a secure signed WebSocket URL
from ElevenLabs, and returns it to the client.

The browser connects directly to ElevenLabs over WebSocket, eliminating
intermediate STT/TTS models, Pipecat WebRTC transports, and server-side audio relays.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import httpx

from backend.app.services.user_memory_service import load_memory_block

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_INSTRUCTION = (
    "You are a helpful voice assistant. Respond briefly and conversationally. "
    "Avoid emojis, bullet points, and any formatting that cannot be spoken aloud."
)


class VoiceConfigurationError(ValueError):
    """Raised when the voice session cannot start due to missing configuration."""


@dataclass
class VoiceSettings:
    agent_id: str | None = None
    elevenlabs_api_key: str | None = None
    user_id: str | None = None
    system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION


async def apply_memory_to_voice_settings(settings: VoiceSettings) -> None:
    """Fold the user's persistent memory into the voice system instruction."""
    if not settings.user_id:
        return
    block = await load_memory_block(settings.user_id)
    if block:
        settings.system_instruction = f"{settings.system_instruction}\n\n{block}"


async def run_voice_pipeline(*args, **kwargs) -> None:
    """Deprecated legacy stub kept for backwards-compatibility with old tests."""
    pass


async def resolve_elevenlabs_credentials(
    explicit_key: Optional[str] = None,
    explicit_agent_id: Optional[str] = None,
) -> tuple[str, str]:
    """Resolve ElevenLabs API key and Agent ID: explicit param > KeyVault > env var."""
    api_key = explicit_key.strip() if explicit_key and explicit_key.strip() else None
    agent_id = explicit_agent_id.strip() if explicit_agent_id and explicit_agent_id.strip() else None

    if not api_key or not agent_id:
        try:
            from backend.app.kernel.keyvault import get_key_vault

            vault = get_key_vault()
            if not api_key:
                for kind in ("voice", "tts", "stt"):
                    val = await vault.get_credential(kind, "elevenlabs")
                    if val:
                        api_key = val
                        break

            if not agent_id:
                val = await vault.get_credential("voice", "elevenlabs_agent_id")
                if val:
                    agent_id = val
                else:
                    rows = await vault.list_credentials()
                    for r in rows:
                        if r.get("provider") == "elevenlabs" and r.get("label"):
                            lbl = r["label"].strip()
                            if lbl.startswith("agent_") or len(lbl) > 10:
                                agent_id = lbl
                                break
        except Exception:
            logger.warning("Key vault lookup for ElevenLabs credentials failed", exc_info=True)

    if not api_key:
        api_key = os.getenv("ELEVENLABS_API_KEY")

    if not agent_id:
        agent_id = os.getenv("ELEVENLABS_AGENT_ID")

    if not api_key or not api_key.strip():
        raise VoiceConfigurationError(
            "ElevenLabs API key is required for voice mode. "
            "Save your key in Settings → Providers or set ELEVENLABS_API_KEY."
        )

    if not agent_id or not agent_id.strip():
        raise VoiceConfigurationError(
            "ElevenLabs Agent ID is required for voice mode. "
            "Save your Agent ID in Settings → Providers or set ELEVENLABS_AGENT_ID."
        )

    return api_key.strip(), _normalize_agent_id(agent_id.strip())


def _normalize_agent_id(raw: str) -> str:
    """Strip common prefixes users paste from ElevenLabs dashboard."""
    cleaned = raw.strip()
    if cleaned.startswith("agent:"):
        cleaned = cleaned[len("agent:") :]
    return cleaned.strip()


async def get_convai_signed_url(agent_id: str, api_key: str) -> str:
    """Request a short-lived signed WebSocket URL from ElevenLabs Conversational AI."""
    url = f"https://api.elevenlabs.io/v1/convai/conversation/get-signed-url?agent_id={agent_id}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers={"xi-api-key": api_key})
            if resp.status_code == 401:
                raise VoiceConfigurationError("Invalid ElevenLabs API key.")
            if resp.status_code == 404:
                raise VoiceConfigurationError(f"ElevenLabs Agent '{agent_id}' not found.")
            if resp.status_code != 200:
                logger.error("ElevenLabs signed URL request failed: %d %s", resp.status_code, resp.text)
                raise VoiceConfigurationError(f"ElevenLabs error ({resp.status_code}): {resp.text}")
            data = resp.json()
            signed_url = data.get("signed_url")
            if not signed_url:
                raise VoiceConfigurationError("ElevenLabs response did not contain a signed_url.")
            return signed_url
    except httpx.RequestError as exc:
        logger.exception("Failed to connect to ElevenLabs API")
        raise VoiceConfigurationError(f"Could not reach ElevenLabs: {exc}") from exc
