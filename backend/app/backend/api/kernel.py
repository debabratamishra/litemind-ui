"""Kernel endpoints: provider status per capability."""

from fastapi import APIRouter

from backend.app.kernel.router import list_provider_status

router = APIRouter(prefix="/api/kernel", tags=["kernel"])


@router.get("/providers")
async def get_providers():
    """Per-capability provider status; ``configured`` means a key/endpoint is
    present (vault or env), NOT live-validated."""
    return await list_provider_status()
