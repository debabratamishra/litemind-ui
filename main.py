
"""Complete FastAPI backend for LLM WebUI with chat and RAG capabilities."""
from contextlib import asynccontextmanager
import asyncio
import json
import logging
import os
import signal
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Optional, List, Dict
import hashlib

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.services.ollama import stream_ollama
from app.services.rag_service import RAGService, CrewAIRAGOrchestrator
from app.services.speech_service import get_speech_service
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from sentence_transformers import SentenceTransformer
from app.services.vllm_service import vllm_service

try:
    import torch
except Exception:
    torch = None

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
# Import and setup logging configuration
try:
    from logging_config import setup_logging, get_logger
    setup_logging()
    logger = get_logger(__name__)
except ImportError:
    # Fallback to basic logging if logging_config is not available
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration and Constants
# ---------------------------------------------------------------------------
from config import Config

# Apply performance settings early
Config.apply_performance_settings()

# Get dynamic configuration
dynamic_config = Config.get_dynamic_config()
UPLOAD_FOLDER = Path(dynamic_config["upload_dir"])

# Handle storage directory - use fallback if not provided by host service manager
storage_dir = dynamic_config.get("storage_dir")
if not storage_dir:
    # Fallback to Config class method
    storage_dir = Config.get_storage_path()
CONFIG_PATH = Path(storage_dir) / "rag_config.json"

# Ensure directories exist
Config.ensure_directories()

DEFAULT_RAG_CONFIG: Dict[str, object] = {
    "provider": "huggingface",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size": 500,
}

def load_rag_config() -> Dict[str, object]:
    try:
        if CONFIG_PATH.exists():
            return json.loads(CONFIG_PATH.read_text())
    except Exception:
        pass
    return dict(DEFAULT_RAG_CONFIG)

def save_rag_config_local(cfg: Dict[str, object]) -> None:
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    except Exception as e:
        logger.warning(f"Failed to persist RAG config: {e}")

rag_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager to handle startup and shutdown."""
    # Record startup time for health checks
    app.state.start_time = time.time()
    logger.info("LLM WebUI API starting up…")
    
    # Log configuration info
    config_info = Config.get_dynamic_config()
    logger.info("Environment: %s", "containerized" if config_info["is_containerized"] else "native")
    logger.info("Upload folder: %s", UPLOAD_FOLDER)
    logger.info("ChromaDB path: %s", config_info["chroma_db_dir"])
    logger.info("Storage path: %s", config_info["storage_dir"])

    # Validate configuration
    validation = Config.validate_configuration()
    if not validation["valid"]:
        logger.error("Configuration validation failed: %s", validation["errors"])
        for error in validation["errors"]:
            logger.error("  - %s", error)
    for warning in validation["warnings"]:
        logger.warning("  - %s", warning)

    # Purge uploads at startup (fresh state on every restart)
    try:
        if UPLOAD_FOLDER.exists():
            shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
        UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        # Set proper permissions for container environment
        if config_info["is_containerized"]:
            os.chmod(UPLOAD_FOLDER, 0o755)
        logger.info("Uploads folder cleared at startup")
    except Exception as e:
        logger.warning("Failed to clear uploads at startup: %s", e)

    # Initialize services
    global rag_service
    rag_service = RAGService()
    logger.info("RAG service ready")

    # Restore RAG config (do not re-index; we start fresh each time)
    try:
        cfg = load_rag_config()
        provider = str(cfg.get("provider", "huggingface")).lower()
        model_name = str(cfg.get("embedding_model", DEFAULT_RAG_CONFIG["embedding_model"]))
        chunk_size_cfg = int(cfg.get("chunk_size", DEFAULT_RAG_CONFIG["chunk_size"]))

        if provider == "ollama":
            rag_service.embedding_function = OllamaEmbeddingFunction(
                model_name=model_name,
                url=f"{config_info['ollama_url']}/api/embeddings",
            )
        else:
            rag_service.embedding_function = LocalHFEmbeddingFunction(
                model_name=model_name,
                batch_size=64,
            )
        rag_service.default_chunk_size = chunk_size_cfg
        logger.info("RAG config restored (fresh store): provider=%s model=%s chunk_size=%s", provider, model_name, chunk_size_cfg)
    except Exception as e:
        logger.warning("Failed to restore RAG config on startup: %s", e)

    # Performance tuning: thread counts for CPU-bound ops
    try:
        cpu_threads = max(1, (os.cpu_count() or 4) - 1)
        os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))
        os.environ.setdefault("NUMEXPR_NUM_THREADS", str(cpu_threads))
        if torch is not None:
            try:
                torch.set_num_threads(cpu_threads)
                if hasattr(torch, "set_num_interop_threads"):
                    torch.set_num_interop_threads(max(1, cpu_threads // 2))
                logger.info("Torch threads set: intra=%s", cpu_threads)
            except Exception as e:
                logger.warning("Could not set Torch threads: %s", e)
        logger.info("OMP/MKL threads set to %s", cpu_threads)
    except Exception as e:
        logger.warning("Thread tuning skipped: %s", e)

    logger.info("Chat service ready")

    yield

    try:
        if 'rag_service' in globals() and rag_service:
            await rag_service.reset_system()
            logger.info("Cleared vector storage on shutdown")
    except Exception as e:
        logger.warning("Failed clearing vector storage on shutdown: %s", e)

    try:
        if UPLOAD_FOLDER.exists():
            shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
            logger.info("Uploads folder cleared on shutdown")
    except Exception as e:
        logger.warning("Failed to clear uploads on shutdown: %s", e)

    logger.info("LLM WebUI API shutting down…")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="LLM WebUI API",
    description="Complete API for LLM WebUI application with Chat and RAG capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates setup
try:
    templates = Jinja2Templates(directory="app/templates")
except Exception:
    templates = None


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# Add these new models
class VLLMTokenRequest(BaseModel):
    token: str

class VLLMModelRequest(BaseModel):
    model_name: str
    dtype: Optional[str] = "auto"
    max_model_len: Optional[int] = None
    gpu_memory_utilization: Optional[float] = 0.9

class ChatRequestEnhanced(BaseModel):
    message: str
    model: Optional[str] = "default"
    temperature: Optional[float] = 0.7
    backend: Optional[str] = "ollama"  # "ollama" or "vllm"
    hf_token: Optional[str] = None

class RAGQueryRequestEnhanced(BaseModel):
    query: str
    messages: Optional[List[dict]] = []
    model: Optional[str] = "default"
    system_prompt: Optional[str] = "You are a helpful assistant."
    n_results: Optional[int] = 3
    use_multi_agent: Optional[bool] = False
    use_hybrid_search: Optional[bool] = False
    backend: Optional[str] = "ollama"  # "ollama" or "vllm"
    hf_token: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    model: str

class RAGConfigRequest(BaseModel):
    provider: str
    embedding_model: str
    chunk_size: int

class STTRequest(BaseModel):
    """Request model for speech-to-text transcription."""
    audio_data: str  # Base64 encoded audio data
    sample_rate: Optional[int] = 16000


# ---------------------------------------------------------------------------
# Health & models
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    """Lightweight health check endpoint."""
    return {"status": "healthy", "service": "LLM WebUI API"}


@app.get("/health/ready")
async def readiness_check():
    """
    Container readiness check endpoint.
    
    This endpoint performs comprehensive checks to determine if the container
    is ready to serve requests. It validates:
    - RAG service initialization
    - Host service connectivity
    - Volume mount accessibility
    - Critical configuration
    """
    try:
        readiness_status = {
            "status": "ready",
            "timestamp": time.time(),
            "checks": {}
        }
        
        # Check RAG service
        if rag_service is None:
            readiness_status["checks"]["rag_service"] = {
                "status": "failed",
                "error": "RAG service not initialized"
            }
            readiness_status["status"] = "not_ready"
        else:
            readiness_status["checks"]["rag_service"] = {
                "status": "ready",
                "details": "RAG service initialized"
            }
        
        # Check critical directories using environment-aware paths
        try:
            from app.services.host_service_manager import host_service_manager
            chroma_path = host_service_manager.environment_config.chroma_db_dir
        except ImportError:
            chroma_path = Path(Config.get_chroma_db_path())
        
        critical_dirs = [UPLOAD_FOLDER, chroma_path]
        dir_checks = {}
        
        for dir_path in critical_dirs:
            dir_name = dir_path.name
            if dir_path.exists() and os.access(dir_path, os.R_OK | os.W_OK):
                dir_checks[dir_name] = {"status": "ready", "path": str(dir_path)}
            else:
                dir_checks[dir_name] = {
                    "status": "failed", 
                    "path": str(dir_path),
                    "error": "Directory not accessible"
                }
                readiness_status["status"] = "not_ready"
        
        readiness_status["checks"]["directories"] = dir_checks
        
        # Check host services (non-blocking)
        try:
            from app.services.host_service_manager import host_service_manager
            
            # Quick service check with short timeout
            ollama_status = await host_service_manager.validate_service_connectivity(
                "Ollama", host_service_manager.environment_config.ollama_url, timeout=2.0
            )
            
            readiness_status["checks"]["ollama"] = {
                "status": "ready" if ollama_status.is_available else "degraded",
                "url": ollama_status.url,
                "response_time_ms": ollama_status.response_time_ms,
                "error": ollama_status.error_message
            }
            
            # vLLM is optional, so don't fail readiness if it's unavailable
            vllm_status = await host_service_manager.validate_service_connectivity(
                "vLLM", host_service_manager.environment_config.vllm_url, timeout=2.0
            )
            
            readiness_status["checks"]["vllm"] = {
                "status": "ready" if vllm_status.is_available else "unavailable",
                "url": vllm_status.url,
                "response_time_ms": vllm_status.response_time_ms,
                "error": vllm_status.error_message,
                "optional": True
            }
            
        except ImportError:
            readiness_status["checks"]["host_services"] = {
                "status": "unavailable",
                "error": "Host service manager not available"
            }
        
        # Return appropriate HTTP status
        if readiness_status["status"] == "ready":
            return readiness_status
        else:
            return JSONResponse(
                status_code=503,
                content=readiness_status
            )
            
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "timestamp": time.time(),
                "error": str(e)
            }
        )


@app.get("/health/live")
async def liveness_check():
    """
    Container liveness check endpoint.
    
    This endpoint performs basic checks to determine if the container
    is alive and should not be restarted. It only checks critical
    application components that indicate the process is functioning.
    """
    try:
        # Basic application health indicators
        liveness_status = {
            "status": "alive",
            "timestamp": time.time(),
            "process_id": os.getpid(),
            "checks": {}
        }
        
        # Check if FastAPI app is responsive
        liveness_status["checks"]["fastapi"] = {
            "status": "alive",
            "details": "FastAPI application responding"
        }
        
        # Check if we can access the file system
        try:
            temp_file = Path("/tmp/liveness_check")
            temp_file.write_text("test")
            temp_file.unlink()
            liveness_status["checks"]["filesystem"] = {
                "status": "alive",
                "details": "Filesystem accessible"
            }
        except Exception as e:
            liveness_status["checks"]["filesystem"] = {
                "status": "failed",
                "error": str(e)
            }
            liveness_status["status"] = "unhealthy"
        
        # Check memory usage (basic check)
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            liveness_status["checks"]["memory"] = {
                "status": "alive" if memory_percent < 95 else "critical",
                "usage_percent": memory_percent
            }
            if memory_percent >= 95:
                liveness_status["status"] = "critical"
        except ImportError:
            # psutil not available, skip memory check
            pass
        
        # Return appropriate status
        if liveness_status["status"] in ["alive", "critical"]:
            return liveness_status
        else:
            return JSONResponse(
                status_code=503,
                content=liveness_status
            )
            
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "dead",
                "timestamp": time.time(),
                "error": str(e)
            }
        )


@app.get("/health/startup")
async def startup_check():
    """
    Container startup check endpoint.
    
    This endpoint indicates when the container has completed its startup
    sequence and is ready to begin serving traffic. It's more comprehensive
    than readiness but less frequent than liveness checks.
    """
    try:
        startup_status = {
            "status": "started",
            "timestamp": time.time(),
            "startup_time": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else None,
            "checks": {}
        }
        
        # Check if all services are initialized
        services_ready = True
        
        # RAG service check
        if rag_service is None:
            startup_status["checks"]["rag_service"] = {
                "status": "initializing",
                "error": "RAG service not yet initialized"
            }
            services_ready = False
        else:
            startup_status["checks"]["rag_service"] = {
                "status": "started",
                "details": "RAG service initialized and ready"
            }
        
        # Configuration validation
        try:
            config_validation = Config.validate_configuration()
            if config_validation["valid"]:
                startup_status["checks"]["configuration"] = {
                    "status": "started",
                    "details": "Configuration validated successfully"
                }
            else:
                startup_status["checks"]["configuration"] = {
                    "status": "failed",
                    "errors": config_validation["errors"],
                    "warnings": config_validation["warnings"]
                }
                services_ready = False
        except Exception as e:
            startup_status["checks"]["configuration"] = {
                "status": "failed",
                "error": str(e)
            }
            services_ready = False
        
        # Directory initialization check
        try:
            directory_status = Config.ensure_directories()
            failed_dirs = [name for name, success in directory_status.items() if not success]
            
            if not failed_dirs:
                startup_status["checks"]["directories"] = {
                    "status": "started",
                    "details": "All directories created successfully"
                }
            else:
                startup_status["checks"]["directories"] = {
                    "status": "failed",
                    "failed_directories": failed_dirs
                }
                services_ready = False
        except Exception as e:
            startup_status["checks"]["directories"] = {
                "status": "failed",
                "error": str(e)
            }
            services_ready = False
        
        # Overall startup status
        if services_ready:
            startup_status["status"] = "started"
            return startup_status
        else:
            startup_status["status"] = "starting"
            return JSONResponse(
                status_code=503,
                content=startup_status
            )
            
    except Exception as e:
        logger.error(f"Startup check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "failed",
                "timestamp": time.time(),
                "error": str(e)
            }
        )


@app.get("/api/system/status")
async def get_system_status():
    """Get comprehensive system status including environment and service information."""
    try:
        from app.services.host_service_manager import host_service_manager
        return await host_service_manager.get_system_status()
    except ImportError:
        return {
            "error": "Host service manager not available",
            "environment": {"is_containerized": False}
        }


@app.get("/api/system/services")
async def validate_host_services():
    """Validate connectivity to all required host services."""
    try:
        from app.services.host_service_manager import host_service_manager
        services = await host_service_manager.validate_all_required_services()
        return {
            "services": {
                name: {
                    "available": status.is_available,
                    "url": status.url,
                    "error": status.error_message,
                    "response_time_ms": status.response_time_ms
                }
                for name, status in services.items()
            }
        }
    except ImportError:
        return {"error": "Host service manager not available"}


@app.get("/api/system/config")
async def get_dynamic_config():
    """Get dynamic configuration based on execution environment."""
    try:
        from app.services.host_service_manager import host_service_manager
        return host_service_manager.get_dynamic_config()
    except ImportError:
        from config import Config
        return Config.get_dynamic_config()


@app.get("/models")
async def get_available_models():
    """Return available Ollama model names from the local backend."""
    try:
        # Get dynamic Ollama URL
        try:
            from app.services.host_service_manager import host_service_manager
            ollama_url = host_service_manager.environment_config.ollama_url
        except ImportError:
            from config import Config
            ollama_url = Config.OLLAMA_API_URL
        
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{ollama_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            model_names = [model["name"] for model in data.get("models", [])]
            return {"models": model_names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not fetch models from Ollama: {str(e)}")


# ---------------------------------------------------------------------------
# Chat routes
# ---------------------------------------------------------------------------

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Serve the chat HTML page if templates are configured."""
    if templates:
        return templates.TemplateResponse("chat.html", {"request": request})
    return HTMLResponse("<h1>Templates not configured</h1>")


# ---------------------------------------------------------------------------
# RAG routes
# ---------------------------------------------------------------------------
@app.get("/rag", response_class=HTMLResponse)
async def rag_page(request: Request):
    """Serve the RAG HTML page if templates are configured."""
    if templates:
        return templates.TemplateResponse("rag.html", {"request": request})
    return HTMLResponse("<h1>Templates not configured</h1>")

@app.get("/api/rag/documents")
async def get_uploaded_documents():
    """List filenames of uploaded and indexed documents."""
    try:
        documents = [f.name for f in UPLOAD_FOLDER.iterdir() if f.is_file()]
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rag/save_config")
async def save_rag_config(request: RAGConfigRequest):
    """Persist RAG configuration (embedding provider/model and chunk size) and update the embedding function without wiping vectors."""
    try:
        global rag_service
        if not rag_service:
            raise HTTPException(status_code=503, detail="RAG service not initialized")

        # Persist new config
        cfg = load_rag_config()
        cfg["provider"] = request.provider
        cfg["embedding_model"] = request.embedding_model
        cfg["chunk_size"] = int(request.chunk_size)
        save_rag_config_local(cfg)

        # Apply embedding function without wiping existing vectors
        if request.provider.lower() == "ollama":
            # Get current dynamic config for Ollama URL
            current_config = Config.get_dynamic_config()
            rag_service.embedding_function = OllamaEmbeddingFunction(
                model_name=request.embedding_model,
                url=f"{current_config['ollama_url']}/api/embeddings",
            )
        elif request.provider.lower() == "huggingface":
            rag_service.embedding_function = LocalHFEmbeddingFunction(
                model_name=request.embedding_model,
                batch_size=64,
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid provider")

        rag_service.default_chunk_size = int(request.chunk_size)

        # Do NOT call recreate_collection here; keep existing KB intact.
        return {"message": "RAG configuration saved successfully", "status": "success"}

    except Exception as e:
        logger.error("Error saving RAG config: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to save RAG configuration: {str(e)}")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
async def process_llm_request(message: str, model: str, temperature: float) -> str:
    """Send a single LLM request and concatenate the streamed response chunks."""
    messages = [{"role": "user", "content": message}]
    response = ""
    async for chunk in stream_ollama(messages, model=model, temperature=temperature):
        response += chunk
    return response


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Return a JSON 404 payload for unknown endpoints."""
    return JSONResponse(status_code=404, content={"error": "Endpoint not found", "path": str(request.url.path)})


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Return a JSON 500 payload for unhandled exceptions."""
    return JSONResponse(status_code=500, content={"error": "Internal server error", "detail": str(exc)})


# ---------------------------------------------------------------------------
# Local embedding function
# ---------------------------------------------------------------------------
class LocalHFEmbeddingFunction:
    """Fast local embedding function with batching and device selection.

    Compatible with Chroma's embedding_function interface.
    """

    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 64):
        if device is None:
            if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    def __call__(self, input):
        """Encode documents to normalized embeddings.
        Chroma 0.4.16+ calls EmbeddingFunction with signature __call__(input=Documents).
        This method accepts either a list of strings or a small mapping containing
        one of: 'input', 'texts', or 'documents'.
        """
        texts = input
        if isinstance(input, dict):
            texts = input.get("input") or input.get("texts") or input.get("documents") or []
        if not isinstance(texts, list):
            texts = [str(texts)]

        embs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embs.tolist()


# ---------------------------------------------------------------------------
# Embedding configuration endpoint
# ---------------------------------------------------------------------------
@app.post("/api/rag/set_embedding")
async def set_embedding(request: dict):
    """Configure the embedding provider/model and rebuild the collection, streaming progress."""
    provider = request.get("provider")
    model_name = request.get("model_name")
    if not provider or not model_name:
        raise HTTPException(status_code=400, detail="Provider and model_name are required")

    async def progress_generator():
        try:
            if provider.lower() == "ollama":
                yield "Checking for Ollama model...\n"
                # Get current dynamic config for Ollama URL
                current_config = Config.get_dynamic_config()
                ollama_url = current_config['ollama_url']
                async with httpx.AsyncClient() as client:
                    tags = await client.get(f"{ollama_url}/api/tags")
                    tags.raise_for_status()
                    models = [m["name"] for m in tags.json().get("models", [])]

                    if model_name not in models:
                        yield f"Model not found. Pulling '{model_name}' from Ollama...\n"
                        pull_url = f"{ollama_url}/api/pull"
                        async with client.stream("POST", pull_url, json={"name": model_name, "stream": True}) as resp:
                            async for line in resp.aiter_lines():
                                if not line:
                                    continue
                                try:
                                    data = json.loads(line)
                                    if "status" in data:
                                        yield f"{data['status']} ({data.get('completed', 0)}/{data.get('total', 0)})\n"
                                except Exception:
                                    pass
                        yield "Ollama model pulled successfully.\n"
                    else:
                        yield "Ollama model already available.\n"

                rag_service.embedding_function = OllamaEmbeddingFunction(
                    url=f"{ollama_url}/api/embeddings",
                    model_name=model_name,
                )
                yield "Ollama embedding function set.\n"

            elif provider.lower() == "huggingface":
                yield f"Loading HuggingFace model: {model_name}\n"
                _tmp = SentenceTransformer(model_name)
                device = "cuda" if (torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu"
                rag_service.embedding_function = LocalHFEmbeddingFunction(
                    model_name=model_name,
                    device=device,
                    batch_size=64,
                )
                yield f"HuggingFace embedding function set on {device}.\n"

            else:
                raise ValueError("Invalid provider")

            # Keep existing collection; if you really want to rebuild, expose a separate reset endpoint.
            yield "Embedding model set. Existing vectors preserved.\n"
        except Exception as e:
            yield f"Error: {str(e)}\n"

    return StreamingResponse(progress_generator(), media_type="text/plain")


@app.post("/api/rag/upload")
async def rag_upload(files: List[UploadFile] = File(...), chunk_size: int = Form(500)):
    """Upload one or more files and queue background indexing into the RAG store with duplicate detection."""
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    results = []
    saved_paths = []
    
    # Check for duplicates before saving files
    for up in files:
        # Check if file is already processed
        is_duplicate, reason = rag_service._is_file_already_processed("", up.filename)
        
        if is_duplicate:
            results.append({
                "filename": up.filename,
                "status": "duplicate",
                "message": reason,
                "chunks_created": 0
            })
            continue
        
        # Save the file
        dest = UPLOAD_FOLDER / up.filename
        with open(dest, "wb") as f:
            f.write(await up.read())
        saved_paths.append((dest, up.filename))

    # Process non-duplicate files
    async def _index():
        sem = asyncio.Semaphore(2)

        async def index_one(path_info):
            async with sem:
                p, filename = path_info
                try:
                    result = await rag_service.add_document(str(p), filename, chunk_size=int(chunk_size))
                    if result:
                        results.append({
                            "filename": filename,
                            **result
                        })
                    else:
                        results.append({
                            "filename": filename,
                            "status": "success",
                            "message": f"Successfully processed {filename}",
                            "chunks_created": 0
                        })
                except Exception as e:
                    results.append({
                        "filename": filename,
                        "status": "error",
                        "message": str(e),
                        "chunks_created": 0
                    })

        if saved_paths:
            await asyncio.gather(*(index_one(path_info) for path_info in saved_paths))

    await _index()  # Wait for processing to complete

    # Prepare summary
    successful = [r for r in results if r["status"] == "success"]
    duplicates = [r for r in results if r["status"] == "duplicate"]
    errors = [r for r in results if r["status"] == "error"]

    return {
        "status": "completed",
        "summary": {
            "total_files": len(files),
            "successful": len(successful),
            "duplicates": len(duplicates),
            "errors": len(errors),
            "total_chunks_created": sum(r.get("chunks_created", 0) for r in successful)
        },
        "results": results
    }


@app.post("/api/rag/reset")
async def reset_rag_system():
    """Reset the entire RAG system by clearing all documents, embeddings, and uploaded files."""
    try:
        global rag_service
        if not rag_service:
            raise HTTPException(status_code=503, detail="RAG service not initialized")

        # Clear uploaded files
        files_removed = 0
        for file_path in UPLOAD_FOLDER.iterdir():
            if file_path.is_file():
                file_path.unlink()
                files_removed += 1
        
        logger.info(f"Removed {files_removed} uploaded files")

        # Reset the RAG service collections and indexes
        await rag_service.reset_system()
        
        return {
            "status": "success",
            "message": f"RAG system reset successfully. Removed {files_removed} files and cleared all embeddings.",
            "files_removed": files_removed
        }

    except Exception as e:
        logger.error(f"Error resetting RAG system: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset RAG system: {str(e)}")


@app.get("/api/rag/status")
async def get_rag_status():
    """Get current RAG system status including document count and collection info."""
    try:
        global rag_service
        if not rag_service:
            return {"status": "not_initialized", "documents": 0, "chunks": 0}

        # Count uploaded files
        uploaded_files = len([f for f in UPLOAD_FOLDER.iterdir() if f.is_file()])

        # Get collection stats
        collection_count = 0
        try:
            if getattr(rag_service, "text_collection", None) is not None:
                collection_count = rag_service.text_collection.count()
        except Exception:
            collection_count = 0

        # Get processed files info
        processed_info = rag_service.get_processed_files_info()

        return {
            "status": "ready",
            "uploaded_files": uploaded_files,
            "indexed_chunks": collection_count,
            "bm25_corpus_size": len(rag_service.bm25_corpus) if rag_service.bm25_corpus else 0,
            "processed_files": processed_info
        }

    except Exception as e:
        logger.error(f"Error getting RAG status: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/api/rag/files")
async def get_processed_files():
    """Get detailed information about all processed files."""
    try:
        global rag_service
        if not rag_service:
            raise HTTPException(status_code=503, detail="RAG service not initialized")

        return rag_service.get_processed_files_info()

    except Exception as e:
        logger.error(f"Error getting processed files: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get processed files: {str(e)}")


@app.delete("/api/rag/files/{filename}")
async def remove_processed_file(filename: str):
    """Remove a specific file from the RAG system."""
    try:
        global rag_service
        if not rag_service:
            raise HTTPException(status_code=503, detail="RAG service not initialized")

        success = rag_service.remove_processed_file(filename)
        
        if success:
            # Also remove the physical file if it exists
            file_path = UPLOAD_FOLDER / filename
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Removed physical file: {filename}")
            
            return {
                "status": "success",
                "message": f"Successfully removed file: {filename}"
            }
        else:
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing file {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove file: {str(e)}")


@app.post("/api/rag/check-duplicates")
async def check_file_duplicates(files: List[UploadFile] = File(...)):
    """Check if uploaded files are duplicates without processing them."""
    try:
        global rag_service
        if not rag_service:
            raise HTTPException(status_code=503, detail="RAG service not initialized")

        results = []
        
        for up in files:
            # Save file temporarily to calculate hash
            temp_path = UPLOAD_FOLDER / f"temp_{up.filename}"
            try:
                with open(temp_path, "wb") as f:
                    f.write(await up.read())
                
                is_duplicate, reason = rag_service._is_file_already_processed(str(temp_path), up.filename)
                
                results.append({
                    "filename": up.filename,
                    "is_duplicate": is_duplicate,
                    "reason": reason if is_duplicate else "File is new and can be processed"
                })
                
            finally:
                # Clean up temp file
                if temp_path.exists():
                    temp_path.unlink()

        return {
            "status": "completed",
            "results": results
        }

    except Exception as e:
        logger.error(f"Error checking duplicates: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check duplicates: {str(e)}")


@app.get("/api/processing/capabilities")
async def processing_capabilities():
    """Report backend processing capabilities and status."""
    if not rag_service:
        return {
            "status": "initializing",
            "message": "RAG service is starting up",
            "capabilities": {"enhanced_csv": False, "ocr_available": False, "memory_optimized": True},
        }
    try:
        return rag_service.get_capabilities()
    except Exception as e:
        logger.exception("Failed to get processing capabilities")
        return {
            "status": "degraded",
            "message": f"Could not determine capabilities: {str(e)}",
            "capabilities": {"enhanced_csv": False, "ocr_available": False, "memory_optimized": True},
        }
    

# ---------------------------------------------------------------------------
# Speech-to-Text routes
# ---------------------------------------------------------------------------

@app.post("/api/stt/transcribe")
async def transcribe_audio_endpoint(request: STTRequest):
    """Transcribe audio data to text using the speech service."""
    try:
        import base64
        
        # Decode base64 audio data
        audio_bytes = base64.b64decode(request.audio_data)
        
        # Get speech service and transcribe
        speech_service = get_speech_service()
        transcribed_text = speech_service.transcribe_audio(audio_bytes, request.sample_rate)
        
        if transcribed_text:
            return {
                "status": "success",
                "transcription": transcribed_text,
                "length": len(transcribed_text)
            }
        else:
            return {
                "status": "error",
                "message": "Failed to transcribe audio",
                "transcription": ""
            }
    except Exception as e:
        logger.error(f"STT transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/api/stt/transcribe-file")
async def transcribe_file_endpoint(file: UploadFile = File(...)):
    """Transcribe an uploaded audio file to text."""
    try:
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(await file.read())
            tmp_file_path = tmp_file.name
        
        try:
            # Get speech service and transcribe
            speech_service = get_speech_service()
            transcribed_text = speech_service.transcribe_file(tmp_file_path)
            
            if transcribed_text:
                return {
                    "status": "success",
                    "transcription": transcribed_text,
                    "filename": file.filename,
                    "length": len(transcribed_text)
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to transcribe audio file",
                    "transcription": ""
                }
        finally:
            # Clean up temporary file
            import os
            os.unlink(tmp_file_path)
            
    except Exception as e:
        logger.error(f"STT file transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"File transcription failed: {str(e)}")

@app.get("/api/stt/status")
async def get_stt_status():
    """Get STT service status and model information."""
    try:
        speech_service = get_speech_service()
        return {
            "status": "ready",
            "model_name": speech_service.model_name,
            "device": "cuda" if hasattr(speech_service.pipe, 'device') and 'cuda' in str(speech_service.pipe.device) else "cpu"
        }
    except Exception as e:
        logger.error(f"STT status check error: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


# --------------------------------------------------------------------------- 
# vLLM routes
# ---------------------------------------------------------------------------

@app.post("/api/vllm/set-token")
async def set_vllm_token(request: VLLMTokenRequest):
    """Set Huggingface token for vLLM."""
    result = vllm_service.set_hf_token(request.token)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result

@app.get("/api/vllm/models")
async def get_vllm_models():
    """Get available vLLM models."""
    return vllm_service.get_available_models()

@app.post("/api/vllm/download-model")
async def download_vllm_model(request: VLLMModelRequest):
    """Download a model from Huggingface."""
    result = await vllm_service.download_model(request.model_name)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result

@app.post("/api/vllm/start-server")
async def start_vllm_server(request: VLLMModelRequest):
    """Start vLLM server with specified model."""
    kwargs = {}
    if request.dtype:
        kwargs["dtype"] = request.dtype
    if request.max_model_len:
        kwargs["max_model_len"] = request.max_model_len
    if request.gpu_memory_utilization:
        kwargs["gpu_memory_utilization"] = request.gpu_memory_utilization
    
    result = await vllm_service.start_vllm_server(request.model_name, **kwargs)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result

@app.post("/api/vllm/stop-server")
async def stop_vllm_server():
    """Stop vLLM server."""
    return await vllm_service.stop_vllm_server()

@app.get("/api/vllm/server-status")
async def vllm_server_status():
    """Check vLLM server status."""
    is_running = await vllm_service.is_server_running()
    return {
        "running": is_running,
        "current_model": vllm_service.current_model
    }


# Also update your single chat endpoint:
@app.post("/api/chat", response_model=ChatResponse) 
async def chat_endpoint_enhanced(request: ChatRequestEnhanced):
    """Process a single chat message with selected backend - FIXED VERSION."""
    
    logger.info(f"Single chat request - Backend: {request.backend}, Model: {request.model}")
    
    try:
        if request.backend == "vllm":
            # Set token if provided
            if request.hf_token:
                token_result = vllm_service.set_hf_token(request.hf_token)
                if token_result["status"] == "error":
                    raise HTTPException(status_code=400, detail=token_result["message"])
            
            # Check if vLLM server is running
            if not await vllm_service.is_server_running():
                raise HTTPException(status_code=400, detail="vLLM server is not running")
            
            # Use vLLM for non-streaming response
            messages = [{"role": "user", "content": request.message}]
            response_text = ""
            async for chunk in vllm_service.stream_vllm_chat(
                messages=messages, 
                model=request.model,
                temperature=request.temperature
            ):
                response_text += chunk
            
            return ChatResponse(response=response_text, model=request.model)
        else:
            # Use existing Ollama logic
            response = await process_llm_request(request.message, request.model, request.temperature)
            return ChatResponse(response=response, model=request.model)
            
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream_enhanced(request: ChatRequestEnhanced):
    """Stream chat responses from selected backend - FIXED VERSION."""
    
    logger.info(f"Chat request received - Backend: {request.backend}, Model: {request.model}")
    
    async def event_generator():
        try:
            if request.backend == "vllm":
                logger.info("Routing to vLLM backend")
                
                # Set token if provided
                if request.hf_token:
                    token_result = vllm_service.set_hf_token(request.hf_token)
                    if token_result["status"] == "error":
                        yield f"Error: {token_result['message']}\n"
                        return
                
                # Check if vLLM server is running
                if not await vllm_service.is_server_running():
                    yield "Error: vLLM server is not running. Please start the server first.\n"
                    return
                
                # Create messages in OpenAI format
                messages = [{"role": "user", "content": request.message}]
                
                # Stream from vLLM
                logger.info(f"Streaming from vLLM with model: {request.model}")
                async for chunk in vllm_service.stream_vllm_chat(
                    messages=messages,
                    model=request.model,
                    temperature=request.temperature
                ):
                    yield chunk + "\n"
                    
            else:
                logger.info("Routing to Ollama backend")
                
                # Use existing Ollama streaming
                messages = [{"role": "user", "content": request.message}]
                async for chunk in stream_ollama(messages, model=request.model, temperature=request.temperature):
                    yield chunk + "\n"
                    
        except Exception as e:
            logger.error(f"Error in chat streaming: {e}")
            yield f"Error: {str(e)}\n"

    return StreamingResponse(event_generator(), media_type="text/plain")

# Update RAG endpoint to support vLLM
@app.post("/api/rag/query")
async def rag_query_enhanced(request: RAGQueryRequestEnhanced):
    """Query RAG system with selected backend."""
    try:
        if request.backend == "vllm":
            # Set token if provided
            if request.hf_token:
                vllm_service.set_hf_token(request.hf_token)

            # Get context from RAG service (robust to missing text_collection)
            context = ""
            try:
                if request.use_hybrid_search and getattr(rag_service, "bm25_model", None):
                    documents = rag_service.hybrid_search(request.query, request.n_results)
                    context = " ".join(documents)
                else:
                    tc = getattr(rag_service, "text_collection", None)
                    if tc is not None:
                        results = tc.query(query_texts=[request.query], n_results=request.n_results)
                        if results and results.get('documents'):
                            context = " ".join(results['documents'][0])
            except Exception as e:
                logger.warning("Vector query failed; proceeding with empty context: %s", e)

            # Create messages for vLLM
            messages = request.messages + [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": f"Context: {context}\n\nQuery: {request.query}"}
            ]

            # Stream response from vLLM
            async def event_generator():
                async for chunk in vllm_service.stream_vllm_chat(
                    messages=messages,
                    model=request.model
                ):
                    yield chunk + "\n"

            return StreamingResponse(event_generator(), media_type="text/plain")
        else:
            # Use existing Ollama RAG logic
            async def event_generator():
                async for chunk in rag_service.query(
                    request.query,
                    request.system_prompt,
                    request.messages,
                    request.n_results,
                    request.use_hybrid_search,
                    request.model
                ):
                    yield chunk + "\n"

            return StreamingResponse(event_generator(), media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def run():
    """Run the Uvicorn server with graceful shutdown handling."""
    config = uvicorn.Config(
        "main:app", host="localhost", port=8000, reload=bool(int(os.getenv("RELOAD", "0"))), log_level="info"
    )
    server = uvicorn.Server(config)
    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    def handle_exit(*_args):
        logger.info("Received exit signal.")
        stop_event.set()

    signals = [signal.SIGINT]
    if hasattr(signal, "SIGTERM"):
        signals.append(signal.SIGTERM)
    for sig in signals:
        try:
            loop.add_signal_handler(sig, handle_exit)
        except (NotImplementedError, RuntimeError):
            signal.signal(sig, lambda s, f: stop_event.set())

    def keyboard_watcher():
        try:
            while not stop_event.is_set():
                pass
        except KeyboardInterrupt:
            stop_event.set()

    if sys.platform.startswith("win"):
        threading.Thread(target=keyboard_watcher, daemon=True).start()

    async def _main():
        server_task = loop.create_task(server.serve())
        await stop_event.wait()
        server.should_exit = True
        await server_task

    try:
        loop.run_until_complete(_main())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt caught. Shutting down gracefully…")
    finally:
        logger.info("App closed.")


if __name__ == "__main__":
    run()

