"""Text-to-speech API endpoints."""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from backend.app.services.tts_service import get_tts_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tts", tags=["speech"])


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    use_cache: Optional[bool] = True


@router.post("/synthesize")
async def synthesize_speech(request: TTSRequest):
    """Convert text to speech audio."""
    try:
        logger.info("TTS request received: text_length=%s", len(request.text) if request.text else 0)
        tts_service = get_tts_service()

        if not tts_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="TTS service not available. Please check if required packages are installed.",
            )

        logger.info("TTS service status: %s", tts_service.get_status())
        audio_data, content_type = await tts_service.synthesize(
            request.text,
            request.voice,
            request.use_cache if request.use_cache is not None else True,
        )

        if not audio_data:
            raise HTTPException(status_code=500, detail="Failed to generate speech audio")

        logger.info("TTS synthesis successful: %s bytes, type=%s", len(audio_data), content_type)
        return Response(
            content=audio_data,
            media_type=content_type,
            headers={
                "Content-Disposition": "inline; filename=speech.mp3",
                "Content-Length": str(len(audio_data)),
                "Cache-Control": "public, max-age=3600",
            },
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("TTS synthesis error")
        raise HTTPException(status_code=500, detail="Speech synthesis failed")


@router.get("/voices")
async def get_tts_voices():
    """Get list of available TTS voices."""
    try:
        tts_service = get_tts_service()
        return {"voices": tts_service.get_available_voices(), "default": "en-US-AriaNeural"}
    except Exception:
        logger.exception("Failed to get TTS voices")
        return {"voices": [], "default": None}


@router.get("/status")
async def get_tts_status():
    """Get TTS service status."""
    try:
        tts_service = get_tts_service()
        return tts_service.get_status()
    except Exception:
        logger.exception("Failed to get TTS status")
        return {"available": False, "error": "Failed to retrieve TTS status"}


@router.post("/synthesize-chunk")
async def synthesize_chunk(request: TTSRequest):
    """Synthesize a single text chunk to speech."""
    try:
        tts_service = get_tts_service()

        if not tts_service.is_available():
            raise HTTPException(status_code=503, detail="TTS service not available")

        loop = asyncio.get_running_loop()
        audio_data = await loop.run_in_executor(
            None,
            tts_service.synthesize_text_chunk,
            request.text,
            request.voice,
        )

        if not audio_data:
            raise HTTPException(status_code=500, detail="Failed to generate speech audio")

        content_type = "audio/wav" if audio_data[:4] == b"RIFF" else "audio/mpeg"
        return Response(
            content=audio_data,
            media_type=content_type,
            headers={"Content-Length": str(len(audio_data)), "Cache-Control": "no-cache"},
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("TTS chunk synthesis error")
        raise HTTPException(status_code=500, detail="Speech synthesis failed")
