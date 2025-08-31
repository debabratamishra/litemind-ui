"""
Configuration management for backend
"""
import json
import logging
from pathlib import Path
from typing import Dict

from config import Config

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_RAG_CONFIG = {
    "provider": "huggingface",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size": 500,
}


class BackendConfig:
    """Backend configuration manager"""
    
    def __init__(self):
        self.dynamic_config = Config.get_dynamic_config()
        self.storage_dir = Path(self.dynamic_config.get("storage_dir", Config.get_storage_path()))
        self.config_path = self.storage_dir / "rag_config.json"
        self.upload_folder = Path(self.dynamic_config["upload_dir"])
        
    def load_rag_config(self) -> Dict:
        """Load RAG configuration from file"""
        try:
            if self.config_path.exists():
                return json.loads(self.config_path.read_text())
        except Exception as e:
            logger.warning(f"Failed to load RAG config: {e}")
        
        return dict(DEFAULT_RAG_CONFIG)
    
    def save_rag_config(self, config: Dict) -> bool:
        """Save RAG configuration to file"""
        try:
            self.config_path.write_text(json.dumps(config, indent=2))
            logger.info(f"RAG config saved: {config}")
            return True
        except Exception as e:
            logger.error(f"Failed to save RAG config: {e}")
            return False
    
    def get_ollama_url(self) -> str:
        """Get Ollama URL from configuration"""
        try:
            from app.services.host_service_manager import host_service_manager
            return host_service_manager.environment_config.ollama_url
        except ImportError:
            return Config.OLLAMA_API_URL
    
    def apply_performance_settings(self):
        """Apply performance optimizations"""
        Config.apply_performance_settings()
        
        # Thread optimization
        import os
        try:
            cpu_threads = max(1, (os.cpu_count() or 4) - 1)
            os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
            
            try:
                import torch
                torch.set_num_threads(cpu_threads)
            except ImportError:
                pass
                
            logger.info(f"Thread optimization applied: {cpu_threads} threads")
        except Exception as e:
            logger.warning(f"Thread tuning failed: {e}")


# Global configuration instance
backend_config = BackendConfig()
