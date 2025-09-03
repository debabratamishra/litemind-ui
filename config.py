import os
import platform
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key')
    
    # Database configuration - handle containerized paths
    SQLALCHEMY_DATABASE_URI = os.getenv('SQLALCHEMY_DATABASE_URI', 'sqlite+aiosqlite:///app/litemindui.db')
    
    # Service URLs
    OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', 'http://localhost:11434')
    VLLM_API_URL = os.getenv('VLLM_API_URL', 'http://localhost:8001')
    
    # Directory paths - handle both containerized and native environments
    @classmethod
    def _get_directory_path(cls, env_var: str, default: str) -> str:
        """Get directory path, preferring native paths when not in container."""
        env_value = os.getenv(env_var)
        if env_value and cls._detect_container_environment():
            return env_value
        return default
    
    @classmethod
    def get_chroma_db_path(cls) -> str:
        return cls._get_directory_path('CHROMA_DB_PATH', './chroma_db')
    
    @classmethod
    def get_upload_folder(cls) -> str:
        return cls._get_directory_path('UPLOAD_FOLDER', './uploads')
    
    @classmethod  
    def get_storage_path(cls) -> str:
        return cls._get_directory_path('STORAGE_PATH', './storage')
    
    # Cache directories - OS and container aware
    HF_HOME = os.getenv('HF_HOME', None)
    OLLAMA_MODELS = os.getenv('OLLAMA_MODELS', None)
    
    # Performance tuning
    OMP_NUM_THREADS = int(os.getenv('OMP_NUM_THREADS', max(1, (os.cpu_count() or 4) - 1)))
    MKL_NUM_THREADS = int(os.getenv('MKL_NUM_THREADS', max(1, (os.cpu_count() or 4) - 1)))
    NUMEXPR_NUM_THREADS = int(os.getenv('NUMEXPR_NUM_THREADS', max(1, (os.cpu_count() or 4) - 1)))
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    DEBUG = bool(int(os.getenv('DEBUG', '0')))
    
    @classmethod
    def _detect_container_environment(cls) -> bool:
        """Detect if running inside a Docker container."""
        # Check for common container indicators
        container_indicators = [
            Path('/.dockerenv').exists(),
            Path('/proc/1/cgroup').exists() and 'docker' in Path('/proc/1/cgroup').read_text(errors='ignore'),
            os.getenv('CONTAINER') is not None,
            os.getenv('DOCKER_CONTAINER') is not None
        ]
        return any(container_indicators)
    
    @classmethod
    def _get_os_cache_paths(cls) -> dict:
        """Get OS-specific cache directory paths."""
        system = platform.system().lower()
        home = Path.home()
        
        if system == "windows":
            hf_cache = home / "AppData" / "Local" / "huggingface"
            ollama_cache = home / ".ollama"
        elif system == "darwin":  # macOS
            hf_cache = home / ".cache" / "huggingface"
            ollama_cache = home / ".ollama"
        else:  # Linux and others
            hf_cache = home / ".cache" / "huggingface"
            ollama_cache = home / ".ollama"
        
        return {
            "huggingface_cache": str(hf_cache),
            "ollama_cache": str(ollama_cache)
        }
    
    @classmethod
    def get_cache_directories(cls) -> dict:
        """Get cache directories based on environment (container vs native)."""
        is_containerized = cls._detect_container_environment()
        
        if is_containerized:
            # In container, use mounted paths or environment variables
            hf_cache = cls.HF_HOME or "/root/.cache/huggingface"
            ollama_cache = cls.OLLAMA_MODELS or "/root/.ollama"
        else:
            # Native environment, use OS-specific paths
            os_paths = cls._get_os_cache_paths()
            hf_cache = cls.HF_HOME or os_paths["huggingface_cache"]
            ollama_cache = cls.OLLAMA_MODELS or os_paths["ollama_cache"]
        
        return {
            "huggingface_cache": hf_cache,
            "ollama_cache": ollama_cache,
            "is_containerized": is_containerized
        }
    
    @classmethod
    def get_persistent_directories(cls) -> dict:
        """Get all persistent directory paths for the application."""
        # Handle database path extraction
        db_path = "./"
        if ':///' in cls.SQLALCHEMY_DATABASE_URI:
            db_file_path = cls.SQLALCHEMY_DATABASE_URI.split(':///')[-1]
            if db_file_path and db_file_path != "":
                db_path = str(Path(db_file_path).parent)
        
        return {
            "chroma_db": cls.get_chroma_db_path(),
            "uploads": cls.get_upload_folder(),
            "storage": cls.get_storage_path(),
            "database_dir": db_path,
        }
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist with proper permissions."""
        try:
            from app.services.host_service_manager import host_service_manager
            return host_service_manager.ensure_directories_exist()
        except ImportError:
            # Fallback directory creation for containerized environment
            results = {}
            
            # Get all directory paths
            persistent_dirs = cls.get_persistent_directories()
            cache_dirs = cls.get_cache_directories()
            
            # Additional directories needed
            additional_dirs = [".streamlit"]
            
            # Create persistent directories
            for name, path_str in persistent_dirs.items():
                if name == "database_dir" and path_str in ["./", ""]:
                    results[name] = True  # Current directory always exists
                    continue
                try:
                    directory = Path(path_str)
                    directory.mkdir(parents=True, exist_ok=True)
                    # Set proper permissions for container environment
                    if cls._detect_container_environment():
                        os.chmod(directory, 0o755)
                    results[name] = True
                except Exception as e:
                    results[name] = False
                    print(f"Failed to create directory {path_str}: {e}")
            
            # Create cache directories (only if not in container or if they don't exist)
            for name, path_str in cache_dirs.items():
                if name == "is_containerized":
                    continue
                try:
                    directory = Path(path_str)
                    if not directory.exists():
                        directory.mkdir(parents=True, exist_ok=True)
                        if cls._detect_container_environment():
                            os.chmod(directory, 0o755)
                    results[f"cache_{name}"] = True
                except Exception as e:
                    results[f"cache_{name}"] = False
                    print(f"Failed to create cache directory {path_str}: {e}")
            
            # Create additional directories
            for dir_name in additional_dirs:
                try:
                    directory = Path(dir_name)
                    directory.mkdir(parents=True, exist_ok=True)
                    if cls._detect_container_environment():
                        os.chmod(directory, 0o755)
                    results[dir_name.replace(".", "")] = True
                except Exception as e:
                    results[dir_name.replace(".", "")] = False
                    print(f"Failed to create directory {dir_name}: {e}")
            
            return results
    
    @classmethod
    def get_dynamic_config(cls):
        """Get configuration that adapts to container vs native environment."""
        try:
            from app.services.host_service_manager import host_service_manager
            return host_service_manager.get_dynamic_config()
        except ImportError:
            # Fallback configuration for containerized environment
            is_containerized = cls._detect_container_environment()
            cache_dirs = cls.get_cache_directories()
            persistent_dirs = cls.get_persistent_directories()
            
            # Ensure directories exist
            directory_status = cls.ensure_directories()
            
            return {
                "is_containerized": is_containerized,
                "ollama_url": cls.OLLAMA_API_URL,
                "vllm_url": cls.VLLM_API_URL,
                "hf_cache_dir": cache_dirs["huggingface_cache"],
                "ollama_cache_dir": cache_dirs["ollama_cache"],
                "upload_dir": persistent_dirs["uploads"],
                "chroma_db_dir": persistent_dirs["chroma_db"],
                "storage_dir": persistent_dirs["storage"],
                "database_uri": cls.SQLALCHEMY_DATABASE_URI,
                "cache_directories_created": directory_status,
                "environment": {
                    "platform": platform.system(),
                    "python_version": platform.python_version(),
                    "container_detected": is_containerized,
                    "performance_threads": {
                        "omp": cls.OMP_NUM_THREADS,
                        "mkl": cls.MKL_NUM_THREADS,
                        "numexpr": cls.NUMEXPR_NUM_THREADS
                    }
                }
            }
    
    @classmethod
    def apply_performance_settings(cls):
        """Apply performance settings for the current environment."""
        # Set thread counts for CPU-bound operations
        os.environ.setdefault("OMP_NUM_THREADS", str(cls.OMP_NUM_THREADS))
        os.environ.setdefault("MKL_NUM_THREADS", str(cls.MKL_NUM_THREADS))
        os.environ.setdefault("NUMEXPR_NUM_THREADS", str(cls.NUMEXPR_NUM_THREADS))
        
        # Set cache directories in environment
        cache_dirs = cls.get_cache_directories()
        if cache_dirs["huggingface_cache"]:
            os.environ.setdefault("HF_HOME", cache_dirs["huggingface_cache"])
        if cache_dirs["ollama_cache"]:
            os.environ.setdefault("OLLAMA_MODELS", cache_dirs["ollama_cache"])
    
    @classmethod
    def validate_configuration(cls) -> dict:
        """Validate the current configuration and return status."""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "info": []
        }
        
        # Check directory accessibility
        persistent_dirs = cls.get_persistent_directories()
        for name, path_str in persistent_dirs.items():
            if name == "database_dir" and path_str == "./":
                continue
            path = Path(path_str)
            if not path.exists():
                validation_results["errors"].append(f"Directory does not exist: {path_str}")
                validation_results["valid"] = False
            elif not os.access(path, os.R_OK | os.W_OK):
                validation_results["errors"].append(f"Directory not accessible: {path_str}")
                validation_results["valid"] = False
        
        # Check cache directories
        cache_dirs = cls.get_cache_directories()
        for name, path_str in cache_dirs.items():
            if name == "is_containerized":
                continue
            path = Path(path_str)
            if not path.exists():
                validation_results["warnings"].append(f"Cache directory will be created: {path_str}")
        
        # Environment info
        validation_results["info"].append(f"Container environment: {cls._detect_container_environment()}")
        validation_results["info"].append(f"Platform: {platform.system()}")
        
        return validation_results