"""
Backend API service for FastAPI server communication.
"""

import logging
from typing import Any, Dict, List, Optional

import requests

from ..config import CONNECT_TIMEOUT, FASTAPI_TIMEOUT, FASTAPI_URL, READ_TIMEOUT

logger = logging.getLogger(__name__)


class BackendService:
    """FastAPI backend client"""

    def __init__(self):
        self.base_url = FASTAPI_URL
        self.timeout = FASTAPI_TIMEOUT
        self.connect_timeout = CONNECT_TIMEOUT
        self.read_timeout = READ_TIMEOUT

    def check_health(self) -> bool:
        """Check backend health"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def get_processing_capabilities(self) -> Optional[Dict[str, Any]]:
        """Get available processing capabilities from the backend."""
        try:
            response = requests.get(f"{self.base_url}/api/processing/capabilities", timeout=self.timeout)
            if response.status_code == 200:
                return response.json()
            return None
        except requests.RequestException:
            return None

    def get_available_models(self) -> List[str]:
        """Get list of available models from the backend."""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=self.timeout)
            payload = response.json()
            return payload.get("models", []) or ["default"]
        except requests.RequestException:
            return ["default"]

    def get_enhanced_models(self) -> Dict[str, Any]:
        """Get local + cloud model listing with metadata."""
        try:
            response = requests.get(f"{self.base_url}/models/enhanced", timeout=self.timeout)
            if response.status_code == 200:
                return response.json()
            return {"local_models": [], "cloud_models": []}
        except requests.RequestException:
            return {"local_models": [], "cloud_models": []}

    def transcribe_audio(self, audio_bytes: bytes, sample_rate: int = 16000) -> Optional[str]:
        """Transcribe audio using the backend STT service.

        Args:
            audio_bytes: Raw audio data bytes
            sample_rate: Audio sample rate (default 16000)

        Returns:
            Transcribed text or None if transcription fails
        """
        import base64

        try:
            # Encode audio as base64 for JSON transport
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            response = requests.post(
                f"{self.base_url}/api/stt/transcribe",
                json={"audio_data": audio_b64, "sample_rate": sample_rate},
                timeout=60,
            )
            if response.status_code == 200:
                return response.json().get("transcription")
            logger.error(f"STT Error: {response.status_code}")
            return None
        except requests.RequestException as exc:
            logger.error(f"STT Connection Error: {exc}")
            return None

    def get_stt_status(self) -> Dict[str, Any]:
        """Get STT service status from backend."""
        try:
            response = requests.get(f"{self.base_url}/api/stt/status", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {"available": False, "model_loaded": False}
        except Exception:
            return {"available": False, "model_loaded": False}


# Singleton instance
backend_service = BackendService()
