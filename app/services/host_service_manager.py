"""
Host Service Manager for Docker Integration

This module provides functionality to detect container environment vs native execution,
manage dynamic configuration based on execution environment, and validate connectivity
to required host services (Ollama, vLLM).
"""

import os
import platform
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import asyncio
import httpx
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ServiceStatus:
    """Represents the status of a host service."""
    name: str
    url: str
    is_available: bool
    error_message: Optional[str] = None
    response_time_ms: Optional[float] = None


@dataclass
class EnvironmentConfig:
    """Configuration that adapts based on execution environment."""
    is_containerized: bool
    ollama_url: str
    vllm_url: str
    hf_cache_dir: Path
    ollama_cache_dir: Path
    upload_dir: Path
    chroma_db_dir: Path
    storage_dir: Path


class HostServiceManager:
    """
    Manages host service detection and configuration for Docker integration.
    
    This class handles:
    - Detection of container vs native execution environment
    - Dynamic configuration based on execution environment
    - Validation of required host services (Ollama, vLLM) connectivity
    - OS-independent cache directory management
    """
    
    def __init__(self):
        self.is_containerized = self._detect_container_environment()
        self.environment_config = self._build_environment_config()
        self._service_cache: Dict[str, ServiceStatus] = {}
        
        logger.info(f"Host Service Manager initialized - Containerized: {self.is_containerized}")
        logger.info(f"Environment config: {self.environment_config}")
    
    def _detect_container_environment(self) -> bool:
        """
        Detect if the application is running inside a container.
        
        Uses multiple detection methods for reliability:
        1. Check for /.dockerenv file (Docker-specific)
        2. Check cgroup information
        3. Check environment variables
        4. Check for container-specific mount points
        
        Returns:
            bool: True if running in a container, False otherwise
        """
        # Method 1: Check for Docker-specific file
        if Path("/.dockerenv").exists():
            logger.info("Container detected via /.dockerenv file")
            return True
        
        # Method 2: Check cgroup information
        try:
            with open("/proc/1/cgroup", "r") as f:
                cgroup_content = f.read()
                if "docker" in cgroup_content or "containerd" in cgroup_content:
                    logger.info("Container detected via cgroup information")
                    return True
        except (FileNotFoundError, PermissionError):
            # /proc/1/cgroup might not exist on non-Linux systems
            pass
        
        # Method 3: Check environment variables
        container_env_vars = [
            "DOCKER_CONTAINER",
            "KUBERNETES_SERVICE_HOST",
            "CONTAINER_NAME"
        ]
        
        for env_var in container_env_vars:
            if os.getenv(env_var):
                logger.info(f"Container detected via environment variable: {env_var}")
                return True
        
        # Method 4: Check for container-specific mount points (Linux)
        if platform.system() == "Linux":
            try:
                with open("/proc/mounts", "r") as f:
                    mounts = f.read()
                    if "overlay" in mounts or "aufs" in mounts:
                        logger.info("Container detected via filesystem mounts")
                        return True
            except (FileNotFoundError, PermissionError):
                pass
        
        logger.info("Native execution environment detected")
        return False
    
    def _get_os_specific_cache_paths(self) -> Tuple[Path, Path]:
        """
        Get OS-specific cache directory paths for Huggingface and Ollama.
        
        Returns:
            Tuple[Path, Path]: (huggingface_cache_path, ollama_cache_path)
        """
        system = platform.system().lower()
        
        if system == "darwin":  # macOS
            hf_cache = Path.home() / ".cache" / "huggingface"
            ollama_cache = Path.home() / ".ollama"
        elif system == "linux":
            hf_cache = Path.home() / ".cache" / "huggingface"
            ollama_cache = Path.home() / ".ollama"
        elif system == "windows":
            hf_cache = Path.home() / ".cache" / "huggingface"
            ollama_cache = Path.home() / ".ollama"
        else:
            # Fallback for unknown systems
            hf_cache = Path.home() / ".cache" / "huggingface"
            ollama_cache = Path.home() / ".ollama"
        
        logger.info(f"OS-specific cache paths - HF: {hf_cache}, Ollama: {ollama_cache}")
        return hf_cache, ollama_cache
    
    def _build_environment_config(self) -> EnvironmentConfig:
        """
        Build configuration that adapts based on execution environment.
        
        Returns:
            EnvironmentConfig: Configuration object with environment-specific settings
        """
        # Get OS-specific cache paths
        hf_cache, ollama_cache = self._get_os_specific_cache_paths()
        
        if self.is_containerized:
            # In container: use mounted paths that map to host directories
            # Check for environment variables that might override default paths
            hf_cache_container = os.getenv("HF_HOME", "/root/.cache/huggingface")
            ollama_cache_container = os.getenv("OLLAMA_MODELS", "/root/.ollama")
            
            config = EnvironmentConfig(
                is_containerized=True,
                ollama_url=os.getenv("OLLAMA_API_URL", "http://localhost:11434"),
                vllm_url=os.getenv("VLLM_API_URL", "http://localhost:8001"),
                hf_cache_dir=Path(hf_cache_container),  # Container mount point
                ollama_cache_dir=Path(ollama_cache_container),  # Container mount point
                upload_dir=Path(os.getenv("UPLOAD_FOLDER", "/app/uploads")),  # Container mount point
                chroma_db_dir=Path(os.getenv("CHROMA_DB_PATH", "/app/chroma_db")),  # Container mount point
                storage_dir=Path(os.getenv("STORAGE_PATH", "/app/storage"))  # Container mount point
            )
        else:
            # Native execution: use host system paths
            config = EnvironmentConfig(
                is_containerized=False,
                ollama_url=os.getenv("OLLAMA_API_URL", "http://localhost:11434"),
                vllm_url=os.getenv("VLLM_API_URL", "http://localhost:8001"),
                hf_cache_dir=hf_cache,
                ollama_cache_dir=ollama_cache,
                upload_dir=Path("./uploads"),
                chroma_db_dir=Path("./chroma_db"),
                storage_dir=Path("./storage")
            )
        
        return config
    
    async def validate_service_connectivity(self, service_name: str, url: str, timeout: float = 5.0) -> ServiceStatus:
        """
        Validate connectivity to a host service.
        
        Args:
            service_name: Name of the service (e.g., "Ollama", "vLLM")
            url: Service URL to test
            timeout: Request timeout in seconds
            
        Returns:
            ServiceStatus: Status object with connectivity information
        """
        import time
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                # For Ollama, test the /api/tags endpoint
                if "11434" in url:  # Ollama port
                    test_url = f"{url}/api/tags"
                # For vLLM, test the /v1/models endpoint
                elif "8001" in url:  # vLLM port
                    test_url = f"{url}/v1/models"
                else:
                    # Generic health check
                    test_url = url
                
                response = await client.get(test_url)
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    status = ServiceStatus(
                        name=service_name,
                        url=url,
                        is_available=True,
                        response_time_ms=response_time
                    )
                    logger.info(f"{service_name} service is available at {url} (response time: {response_time:.1f}ms)")
                else:
                    status = ServiceStatus(
                        name=service_name,
                        url=url,
                        is_available=False,
                        error_message=f"HTTP {response.status_code}",
                        response_time_ms=response_time
                    )
                    logger.warning(f"{service_name} service returned HTTP {response.status_code} at {url}")
                
        except httpx.TimeoutException:
            status = ServiceStatus(
                name=service_name,
                url=url,
                is_available=False,
                error_message="Connection timeout"
            )
            logger.warning(f"{service_name} service timeout at {url}")
            
        except httpx.ConnectError:
            status = ServiceStatus(
                name=service_name,
                url=url,
                is_available=False,
                error_message="Connection refused"
            )
            logger.warning(f"{service_name} service connection refused at {url}")
            
        except Exception as e:
            status = ServiceStatus(
                name=service_name,
                url=url,
                is_available=False,
                error_message=str(e)
            )
            logger.error(f"Error checking {service_name} service at {url}: {e}")
        
        # Cache the result
        self._service_cache[service_name] = status
        return status
    
    async def validate_all_required_services(self) -> Dict[str, ServiceStatus]:
        """
        Validate connectivity to all required host services.
        
        Returns:
            Dict[str, ServiceStatus]: Dictionary mapping service names to their status
        """
        services_to_check = [
            ("Ollama", self.environment_config.ollama_url),
            ("vLLM", self.environment_config.vllm_url)
        ]
        
        results = {}
        
        # Check services concurrently
        tasks = [
            self.validate_service_connectivity(name, url)
            for name, url in services_to_check
        ]
        
        statuses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for (name, url), status in zip(services_to_check, statuses):
            if isinstance(status, Exception):
                results[name] = ServiceStatus(
                    name=name,
                    url=url,
                    is_available=False,
                    error_message=str(status)
                )
            else:
                results[name] = status
        
        return results
    
    def get_cached_service_status(self, service_name: str) -> Optional[ServiceStatus]:
        """
        Get cached service status without making a new request.
        
        Args:
            service_name: Name of the service
            
        Returns:
            Optional[ServiceStatus]: Cached status or None if not cached
        """
        return self._service_cache.get(service_name)
    
    def get_host_cache_paths(self) -> Dict[str, str]:
        """
        Get host system cache paths for Docker volume mounting.
        These are the paths on the host system that should be mounted into containers.
        
        Returns:
            Dict[str, str]: Dictionary mapping cache types to host paths
        """
        hf_cache, ollama_cache = self._get_os_specific_cache_paths()
        
        return {
            "huggingface_cache": str(hf_cache),
            "ollama_cache": str(ollama_cache),
            "uploads": str(Path("./uploads").resolve()),
            "chroma_db": str(Path("./chroma_db").resolve()),
            "storage": str(Path("./storage").resolve()),
            "streamlit_config": str(Path("./.streamlit").resolve())
        }
    
    def ensure_directories_exist(self) -> Dict[str, bool]:
        """
        Ensure all required directories exist with proper permissions.
        
        Returns:
            Dict[str, bool]: Dictionary mapping directory names to creation success
        """
        directories = {
            "hf_cache": self.environment_config.hf_cache_dir,
            "ollama_cache": self.environment_config.ollama_cache_dir,
            "uploads": self.environment_config.upload_dir,
            "chroma_db": self.environment_config.chroma_db_dir,
            "storage": self.environment_config.storage_dir
        }
        
        results = {}
        
        for name, path in directories.items():
            try:
                path.mkdir(parents=True, exist_ok=True)
                # Set appropriate permissions (readable/writable by owner, readable by group)
                if hasattr(os, 'chmod'):
                    try:
                        os.chmod(path, 0o755)
                    except (OSError, PermissionError):
                        # Ignore permission errors - directory creation succeeded
                        pass
                results[name] = True
                logger.info(f"Directory ensured: {name} -> {path}")
            except Exception as e:
                results[name] = False
                logger.error(f"Failed to create directory {name} at {path}: {e}")
        
        return results
    
    def ensure_host_cache_directories_exist(self) -> Dict[str, bool]:
        """
        Ensure host cache directories exist before Docker mounting.
        This should be called before starting Docker containers to ensure
        the host directories exist and have proper permissions.
        
        Returns:
            Dict[str, bool]: Dictionary mapping directory names to creation success
        """
        host_paths = self.get_host_cache_paths()
        results = {}
        
        for name, path_str in host_paths.items():
            try:
                path = Path(path_str)
                path.mkdir(parents=True, exist_ok=True)
                # Set appropriate permissions
                if hasattr(os, 'chmod'):
                    try:
                        os.chmod(path, 0o755)
                    except (OSError, PermissionError):
                        # Ignore permission errors - directory creation succeeded
                        pass
                results[name] = True
                logger.info(f"Host directory ensured: {name} -> {path}")
            except Exception as e:
                results[name] = False
                logger.error(f"Failed to create host directory {name} at {path_str}: {e}")
        
        return results
    
    def get_dynamic_config(self) -> Dict[str, Any]:
        """
        Get dynamic configuration dictionary for use by other services.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        return {
            "is_containerized": self.environment_config.is_containerized,
            "ollama_url": self.environment_config.ollama_url,
            "vllm_url": self.environment_config.vllm_url,
            "hf_cache_dir": str(self.environment_config.hf_cache_dir),
            "ollama_cache_dir": str(self.environment_config.ollama_cache_dir),
            "upload_dir": str(self.environment_config.upload_dir),
            "chroma_db_dir": str(self.environment_config.chroma_db_dir),
            "storage_dir": str(self.environment_config.storage_dir),
            "cache_directories_created": self.ensure_directories_exist()
        }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status including environment and service information.
        
        Returns:
            Dict[str, Any]: System status dictionary
        """
        # Validate services
        service_statuses = await self.validate_all_required_services()
        
        # Get directory status
        directory_status = self.ensure_directories_exist()
        
        return {
            "environment": {
                "is_containerized": self.environment_config.is_containerized,
                "platform": platform.system(),
                "python_version": platform.python_version()
            },
            "configuration": self.get_dynamic_config(),
            "services": {
                name: {
                    "available": status.is_available,
                    "url": status.url,
                    "error": status.error_message,
                    "response_time_ms": status.response_time_ms
                }
                for name, status in service_statuses.items()
            },
            "directories": directory_status
        }


# Global instance
host_service_manager = HostServiceManager()