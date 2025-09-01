"""
Backend API service for FastAPI server communication.
"""
import logging
import requests
from typing import Dict, List, Optional, Any, Tuple

from ..config import FASTAPI_URL, FASTAPI_TIMEOUT, CONNECT_TIMEOUT, READ_TIMEOUT

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

    def get_vllm_models(self) -> Dict[str, List[str]]:
        """Get available vLLM models."""
        try:
            response = requests.get(f"{self.base_url}/api/vllm/models", timeout=10)
            if response.status_code == 200:
                return response.json() or {"local_models": [], "popular_models": []}
            return {"local_models": [], "popular_models": []}
        except Exception as e:
            logger.error(f"Error fetching vLLM models: {e}")
            return {"local_models": [], "popular_models": []}

    def get_vllm_server_status(self) -> Dict[str, Any]:
        """Get vLLM server status."""
        try:
            response = requests.get(f"{self.base_url}/api/vllm/server-status", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {"running": False, "current_model": None}
        except Exception:
            return {"running": False, "current_model": None}

    def validate_hf_token(self, token: str) -> bool:
        """Validate Hugging Face token."""
        try:
            response = requests.post(
                f"{self.base_url}/api/vllm/set-token",
                json={"token": token},
                timeout=10,
            )
            return response.status_code == 200
        except Exception:
            return False

    def download_vllm_model(self, model_name: str) -> bool:
        """Download a vLLM model."""
        try:
            response = requests.post(
                f"{self.base_url}/api/vllm/download-model",
                json={"model_name": model_name},
                timeout=300,
            )
            return response.status_code == 200
        except Exception:
            return False

    def start_vllm_server(self, model_name: str, dtype: str = "auto") -> bool:
        """Start vLLM server with specified model."""
        try:
            response = requests.post(
                f"{self.base_url}/api/vllm/start-server",
                json={"model_name": model_name, "dtype": dtype},
                timeout=60,
            )
            return response.status_code == 200
        except Exception:
            return False

    def stop_vllm_server(self) -> bool:
        """Stop vLLM server."""
        try:
            response = requests.post(f"{self.base_url}/api/vllm/stop-server", timeout=10)
            return response.status_code == 200
        except Exception:
            return False


# Singleton instance
backend_service = BackendService()
