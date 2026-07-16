"""WebRTC ICE / NAT configuration for realtime voice chat.

Builds the ICE-server lists for both the browser peer and the in-container
aiortc peer. Credentials are read from the environment so TURN can be
configured without code changes.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


_DEFAULT_STUN_URL = "stun:stun.l.google.com:19302"
_DEFAULT_SERVER_TURN_URL = "turn:host.docker.internal:3478"


def _turn_credentials() -> Tuple[str, str]:
    """Return (username, credential) for the TURN server from the environment."""
    return (
        os.getenv("WEBRTC_TURN_USERNAME", "").strip(),
        os.getenv("WEBRTC_TURN_CREDENTIAL", "").strip(),
    )


def _make_turn_server(turn_url: str) -> Dict[str, Any]:
    """Build a single TURN ICE-server dict, attaching credentials when present."""
    turn_server: Dict[str, Any] = {"urls": [turn_url]}
    turn_user, turn_cred = _turn_credentials()
    if turn_user:
        turn_server["username"] = turn_user
    if turn_cred:
        turn_server["credential"] = turn_cred
    return turn_server


def _parse_ice_override(env_var: str) -> Optional[List[Dict[str, Any]]]:
    """Parse a full JSON ICE-server override from ``env_var`` if set and valid."""
    raw = os.getenv(env_var, "").strip()
    if not raw:
        return None
    try:
        servers = json.loads(raw)
        if isinstance(servers, list) and servers:
            return servers
        logger.warning("%s is not a non-empty JSON list; ignoring", env_var)
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse %s as JSON: %s", env_var, e)
    return None


def _is_in_docker() -> bool:
    """Return True if running inside a Docker container."""
    if os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"):
        return True
    try:
        with open("/proc/1/cgroup", "rt") as f:
            if "docker" in f.read() or "kubepods" in f.read():
                return True
    except Exception:
        pass
    return False


def _build_frontend_ice_servers() -> List[Dict[str, Any]]:
    """ICE servers for the BROWSER peer (STUN + optional host-reachable TURN)."""
    override = _parse_ice_override("WEBRTC_ICE_SERVERS")
    if override is not None:
        return override

    stun_url = os.getenv("WEBRTC_STUN_URL", _DEFAULT_STUN_URL).strip() or _DEFAULT_STUN_URL
    servers: List[Dict[str, Any]] = [{"urls": [stun_url]}]

    # Only include TURN server if running inside Docker
    if _is_in_docker():
        turn_url = os.getenv("WEBRTC_TURN_URL", "").strip()
        if turn_url:
            servers.append(_make_turn_server(turn_url))

    return servers


def _build_server_ice_servers() -> List[Dict[str, Any]]:
    """ICE servers for the in-container SERVER peer.

    When running in Docker, the server peer is given TURN ONLY to avoid
    failing STUN binding requests that crash aioice.
    When running standalone, it falls back to STUN.
    """
    override = _parse_ice_override("WEBRTC_SERVER_ICE_SERVERS")
    if override is not None:
        return override

    if _is_in_docker():
        server_turn_url = os.getenv("WEBRTC_SERVER_TURN_URL", "").strip()
        if not server_turn_url:
            server_turn_url = _DEFAULT_SERVER_TURN_URL
        return [_make_turn_server(server_turn_url)]

    # Standalone mode: just use STUN
    stun_url = os.getenv("WEBRTC_STUN_URL", _DEFAULT_STUN_URL).strip() or _DEFAULT_STUN_URL
    return [{"urls": [stun_url]}]


def _build_frontend_rtc_configuration() -> Dict[str, Any]:
    """RTCConfiguration dict for the browser peer."""
    return {"iceServers": _build_frontend_ice_servers()}


def _build_server_rtc_configuration() -> Dict[str, Any]:
    """RTCConfiguration dict for the in-container aiortc peer."""
    return {"iceServers": _build_server_ice_servers()}
