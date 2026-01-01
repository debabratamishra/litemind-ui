"""
Unified Environment Detection Module.

This module provides a singleton for detecting the runtime environment
(containerized vs native) to eliminate code duplication across the codebase.
"""
import os
import platform
import logging
from pathlib import Path
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)


class EnvironmentDetector:
    """
    Singleton class for environment detection.
    
    Detects whether the application is running inside a container (Docker, Kubernetes, etc.)
    or in a native host environment. Uses multiple detection methods for reliability.
    """
    
    _instance: Optional['EnvironmentDetector'] = None
    _initialized: bool = False
    
    def __new__(cls) -> 'EnvironmentDetector':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not EnvironmentDetector._initialized:
            self._is_containerized = self._detect_container_environment()
            self._platform = platform.system().lower()
            EnvironmentDetector._initialized = True
            logger.info(f"Environment detected - Containerized: {self._is_containerized}, Platform: {self._platform}")
    
    @lru_cache(maxsize=1)
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
            logger.debug("Container detected via /.dockerenv file")
            return True
        
        # Method 2: Check cgroup information
        try:
            cgroup_path = Path("/proc/1/cgroup")
            if cgroup_path.exists():
                cgroup_content = cgroup_path.read_text(errors='ignore')
                if "docker" in cgroup_content or "containerd" in cgroup_content:
                    logger.debug("Container detected via cgroup information")
                    return True
        except (PermissionError, OSError):
            pass
        
        # Method 3: Check environment variables
        container_env_vars = [
            "DOCKER_CONTAINER",
            "CONTAINER",
            "KUBERNETES_SERVICE_HOST",
            "CONTAINER_NAME"
        ]
        
        for env_var in container_env_vars:
            if os.getenv(env_var):
                logger.debug(f"Container detected via environment variable: {env_var}")
                return True
        
        # Method 4: Check for container-specific mount points (Linux)
        if platform.system() == "Linux":
            try:
                mounts_path = Path("/proc/mounts")
                if mounts_path.exists():
                    mounts = mounts_path.read_text(errors='ignore')
                    if "overlay" in mounts or "aufs" in mounts:
                        logger.debug("Container detected via filesystem mounts")
                        return True
            except (PermissionError, OSError):
                pass
        
        logger.debug("Native execution environment detected")
        return False
    
    @property
    def is_containerized(self) -> bool:
        """Check if running in a container environment."""
        return self._is_containerized
    
    @property
    def is_docker(self) -> bool:
        """Alias for is_containerized for backward compatibility."""
        return self._is_containerized
    
    @property
    def platform_name(self) -> str:
        """Get the current platform name (darwin, linux, windows)."""
        return self._platform
    
    @property
    def is_macos(self) -> bool:
        """Check if running on macOS."""
        return self._platform == "darwin"
    
    @property
    def is_linux(self) -> bool:
        """Check if running on Linux."""
        return self._platform == "linux"
    
    @property
    def is_windows(self) -> bool:
        """Check if running on Windows."""
        return self._platform == "windows"
    
    def get_cache_paths(self) -> tuple:
        """
        Get OS-specific cache directory paths for Huggingface and Ollama.
        
        Returns:
            Tuple[Path, Path]: (huggingface_cache_path, ollama_cache_path)
        """
        home = Path.home()
        hf_cache = home / ".cache" / "huggingface"
        ollama_cache = home / ".ollama"
        return hf_cache, ollama_cache


# Singleton instance for easy import
environment = EnvironmentDetector()


# Convenience functions for backward compatibility
def is_containerized() -> bool:
    """Check if running in a container environment."""
    return environment.is_containerized


def is_docker() -> bool:
    """Check if running in Docker."""
    return environment.is_docker


def get_platform() -> str:
    """Get the current platform name."""
    return environment.platform_name
