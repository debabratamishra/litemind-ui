"""Kernel endpoints: provider status per capability."""

from __future__ import annotations

import logging
import time

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app.kernel.router import CAPABILITIES, list_provider_status, resolve_provider_config

router = APIRouter(prefix="/api/kernel", tags=["kernel"])
logger = logging.getLogger(__name__)

_PROBE_CAPABILITIES = ("llm", "vlm", "embeddings", "stt", "tts", "voice")
_PROBE_PATHS = {
    "openrouter": "/models",
    "nvidia_nim": "/models",
    "ollama": "/api/tags",
    "elevenlabs": "/v1/user",
}
_PROBE_TIMEOUT_SECONDS = 10.0


class ProviderTestRequest(BaseModel):
    capability: str


@router.get("/providers")
async def get_providers():
    """Per-capability provider status; ``configured`` means a key/endpoint is
    present (vault or env), NOT live-validated."""
    return await list_provider_status()


@router.post("/providers/test")
async def test_provider(request: ProviderTestRequest) -> dict:
    """Probe the resolved provider's models-style endpoint with the resolved
    config (explicit param > vault > env). Never echoes the key."""
    capability = request.capability.strip()
    if capability not in CAPABILITIES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown capability '{capability}'. Expected one of: {', '.join(CAPABILITIES)}.",
        )
    if capability not in _PROBE_CAPABILITIES:
        raise HTTPException(status_code=501, detail=f"No probe implemented for capability '{capability}'")

    config = await resolve_provider_config(capability)
    probe_path = _PROBE_PATHS.get(config.backend)
    if probe_path is None:
        raise HTTPException(status_code=501, detail=f"No probe implemented for provider '{config.backend}'")

    if config.api_base is None:
        return {"ok": False, "error": f"No API base configured for provider '{config.backend}'"}
    url = f"{config.api_base.rstrip('/')}{probe_path}"
    if config.backend == "elevenlabs":
        headers = {"xi-api-key": config.api_key} if config.api_key else {}
    else:
        headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_SECONDS) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
    except httpx.HTTPError:
        logger.exception("Provider probe failed for capability '%s'", capability)
        return {"ok": False, "error": "Provider probe failed."}
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    return {"ok": True, "latency_ms": latency_ms}
