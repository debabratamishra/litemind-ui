"""Realtime voice signaling endpoint (SDP offer/answer over HTTP).

The browser POSTs an SDP offer; the server answers and runs the Pipecat
pipeline as a background task. All ongoing transcript/control events are sent
back over the WebRTC data channel (see app.services.voice_pipeline).
"""
import logging

from fastapi import APIRouter, BackgroundTasks, Depends
from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection

from app.backend.api.auth_deps import User, get_current_user
from app.services.voice_pipeline import VoiceSettings, run_voice_pipeline

logger = logging.getLogger(__name__)

router = APIRouter()

pcs_map: dict[str, SmallWebRTCConnection] = {}
ice_servers = [IceServer(urls="stun:stun.l.google.com:19302")]


async def run_voice_pipeline_safe(connection: SmallWebRTCConnection, settings: VoiceSettings) -> None:
    """Run the voice pipeline, emitting an ``error`` event if it fails.

    Best-effort: the data channel may already be gone by the time an error is
    surfaced, so every ``send_app_message`` is guarded.
    """
    try:
        await run_voice_pipeline(connection, settings)
    except Exception as exc:  # noqa: BLE001 - surface any pipeline failure to the client
        logger.exception("Voice pipeline failed for pc %s", connection.pc_id)
        try:
            connection.send_app_message({"type": "error", "message": str(exc)})
        except Exception:  # noqa: BLE001 - data channel may be closed
            logger.debug("Could not send error event (data channel closed)")


@router.post("/api/voice/offer")
async def offer(request: dict, background_tasks: BackgroundTasks, user: User = Depends(get_current_user)):
    pc_id = request.get("pc_id")
    if not pc_id:
        return {"error": "pc_id required"}

    if pc_id in pcs_map:
        connection = pcs_map[pc_id]
        await connection.renegotiate(
            sdp=request["sdp"], type=request["type"], restart_pc=request.get("restart_pc", False)
        )
    else:
        connection = SmallWebRTCConnection(ice_servers)
        await connection.initialize(sdp=request["sdp"], type=request["type"])

        @connection.event_handler("closed")
        async def handle_closed(conn: SmallWebRTCConnection):
            logger.info("Discarding peer connection %s", conn.pc_id)
            try:
                conn.send_app_message({"type": "ended"})
            except Exception:  # noqa: BLE001 - data channel may already be gone
                logger.debug("Could not send 'ended' event (data channel closed)")
            pcs_map.pop(conn.pc_id, None)

        settings = VoiceSettings(
            model=request.get("model"),
            backend=request.get("backend"),
            api_key=request.get("api_key"),
            api_base=request.get("api_base"),
            temperature=float(request.get("temperature", 0.7)),
            max_tokens=int(request.get("max_tokens", 512)),
            voice=request.get("voice"),
            system_instruction=request.get("system_instruction")
            or "You are a helpful voice assistant. Respond briefly and conversationally. "
            "Avoid emojis, bullet points, and any formatting that cannot be spoken aloud.",
            user_id=user.id,
        )
        background_tasks.add_task(run_voice_pipeline_safe, connection, settings)

    answer = connection.get_answer()
    pcs_map[answer["pc_id"]] = connection
    return answer
