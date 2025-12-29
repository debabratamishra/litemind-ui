"""
RAG service for handling document processing and retrieval-augmented generation.
"""
import logging
import requests
from typing import Dict, List, Optional, Any, Tuple

from ..config import FASTAPI_URL, FASTAPI_TIMEOUT, CONNECT_TIMEOUT, READ_TIMEOUT

logger = logging.getLogger(__name__)


class RAGService:
    """Service for handling RAG operations."""
    
    def __init__(self):
        self.base_url = FASTAPI_URL
        self.timeout = FASTAPI_TIMEOUT

    def save_configuration(
        self, 
        provider: str, 
        embedding_model: str, 
        chunk_size: int
    ) -> Tuple[bool, str]:
        """Save RAG configuration to the backend."""
        try:
            response = requests.post(
                f"{self.base_url}/api/rag/save_config",
                json={
                    "provider": provider, 
                    "embedding_model": embedding_model, 
                    "chunk_size": chunk_size
                },
                timeout=self.timeout,
            )
            return (
                response.status_code == 200, 
                "Configuration Saved" if response.status_code == 200 else response.text
            )
        except requests.RequestException as exc:
            return False, f"Configuration Error: {exc}"

    def get_status(self) -> Optional[Dict[str, Any]]:
        """Get current RAG system status."""
        try:
            response = requests.get(f"{self.base_url}/api/rag/status", timeout=self.timeout)
            if response.status_code == 200:
                return response.json()
            return None
        except requests.RequestException:
            return None

    def reset_system(self) -> Tuple[bool, str]:
        """Reset the entire RAG system."""
        try:
            response = requests.post(f"{self.base_url}/api/rag/reset", timeout=self.timeout)
            if response.status_code == 200:
                result = response.json()
                return True, result.get("message", "RAG system reset successfully")
            else:
                return False, f"Reset failed with status {response.status_code}: {response.text}"
        except requests.RequestException as exc:
            return False, f"Reset Error: {exc}"

    def check_file_duplicates(self, uploaded_files: List[Any]) -> Optional[Dict[str, Any]]:
        """Check if uploaded files are duplicates."""
        try:
            files = [("files", (uf.name, uf.getvalue(), uf.type)) for uf in uploaded_files]
            
            response = requests.post(
                f"{self.base_url}/api/rag/check-duplicates",
                files=files,
                timeout=self.timeout,
            )
            
            if response.status_code == 200:
                return response.json()
            return None
        except requests.RequestException as exc:
            logger.error(f"Duplicate Check Error: {exc}")
            return None

    def upload_files(self, uploaded_files: List[Any], chunk_size: int = 500) -> Tuple[bool, Dict[str, Any]]:
        """Upload files for enhanced processing."""
        if not uploaded_files:
            return False, {}

        try:
            files = [("files", (uf.name, uf.getvalue(), uf.type)) for uf in uploaded_files]

            response = requests.post(
                f"{self.base_url}/api/rag/upload",
                files=files,
                data={"chunk_size": chunk_size},
                timeout=300,
            )

            if response.status_code == 200:
                return True, response.json()

            logger.error(f"Upload failed with status {response.status_code}: {response.text}")
            return False, {}

        except requests.RequestException as exc:
            logger.error(f"Upload Error: {exc}")
            return False, {}

    def get_processed_files(self) -> Optional[Dict[str, Any]]:
        """Get information about all processed files."""
        try:
            response = requests.get(f"{self.base_url}/api/rag/files", timeout=self.timeout)
            if response.status_code == 200:
                return response.json()
            return None
        except requests.RequestException:
            return None

    def remove_file(self, filename: str) -> Tuple[bool, str]:
        """Remove a specific file from the RAG system."""
        try:
            response = requests.delete(
                f"{self.base_url}/api/rag/files/{filename}", 
                timeout=self.timeout
            )
            if response.status_code == 200:
                result = response.json()
                return True, result.get("message", "File removed successfully")
            else:
                return False, f"Failed to remove file: {response.text}"
        except requests.RequestException as exc:
            return False, f"Remove Error: {exc}"

    def query_rag(
        self,
        query: str,
        messages: List[Dict[str, str]],
        model: str,
        system_prompt: str,
        n_results: int = 3,
        use_multi_agent: bool = False,
        use_hybrid_search: bool = False,
    ) -> Optional[str]:
        """Call the non-streaming RAG endpoint."""
        try:
            response = requests.post(
                f"{self.base_url}/api/rag/query",
                json={
                    "query": query,
                    "messages": messages,
                    "model": model,
                    "system_prompt": system_prompt,
                    "n_results": n_results,
                    "use_multi_agent": use_multi_agent,
                    "use_hybrid_search": use_hybrid_search,
                },
                timeout=self.timeout * 2,
            )
            return response.text if response.status_code == 200 else None
        except requests.RequestException as exc:
            logger.error(f"RAG API Error: {exc}")
            return None

    def stream_rag_query(
        self,
        query: str,
        messages: List[Dict[str, str]],
        model: str,
        system_prompt: str,
        n_results: int = 3,
        use_multi_agent: bool = False,
        use_hybrid_search: bool = False,
        backend: str = "ollama",
        hf_token: Optional[str] = None,
        conversation_summary: Optional[str] = None,
        session_id: Optional[str] = None,
        temperature: float = 0.7,
    ) -> requests.Response:
        """Stream a RAG response from the backend with conversation memory support."""
        payload = {
            "query": query,
            "messages": messages,
            "model": model,
            "system_prompt": system_prompt,
            "n_results": n_results,
            "use_multi_agent": use_multi_agent,
            "use_hybrid_search": use_hybrid_search,
            "temperature": temperature,
        }
        
        if backend == "vllm":
            payload["backend"] = backend
            if hf_token:
                payload["hf_token"] = hf_token

        # Add conversation memory fields
        if conversation_summary:
            payload["conversation_summary"] = conversation_summary
        if session_id:
            payload["session_id"] = session_id

        response = requests.post(
            f"{self.base_url}/api/rag/query",
            json=payload,
            stream=True,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )
        response.raise_for_status()
        return response


# Singleton instance
rag_service = RAGService()
