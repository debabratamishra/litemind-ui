"""
LLM WebUI FastAPI Backend
Production-ready API server with chat and RAG capabilities.
"""
import asyncio
import json
import logging
import os
import signal
import shutil
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.services.ollama import stream_ollama
from app.services.rag_service import RAGService
from app.services.speech_service import get_speech_service
from app.services.vllm_service import vllm_service
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from config import Config
from sentence_transformers import SentenceTransformer

try:
    import torch
except ImportError:
    torch = None

# Configure logging
try:
    from logging_config import get_logger, setup_logging
    setup_logging()
    logger = get_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Configuration
Config.apply_performance_settings()
dynamic_config = Config.get_dynamic_config()
UPLOAD_FOLDER = Path(dynamic_config["upload_dir"])
storage_dir = dynamic_config.get("storage_dir", Config.get_storage_path())
CONFIG_PATH = Path(storage_dir) / "rag_config.json"
Config.ensure_directories()

DEFAULT_RAG_CONFIG = {
    "provider": "huggingface",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size": 500,
}

# Global service instance
rag_service = None


# Request/Response Models
class ChatRequestEnhanced(BaseModel):
    message: str
    model: Optional[str] = "default"
    temperature: Optional[float] = 0.7
    backend: Optional[str] = "ollama"
    hf_token: Optional[str] = None


class RAGQueryRequestEnhanced(BaseModel):
    query: str
    messages: Optional[List[dict]] = []
    model: Optional[str] = "default"
    system_prompt: Optional[str] = "You are a helpful assistant."
    n_results: Optional[int] = 3
    use_multi_agent: Optional[bool] = False
    use_hybrid_search: Optional[bool] = False
    backend: Optional[str] = "ollama"
    hf_token: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    model: str


class RAGConfigRequest(BaseModel):
    provider: str
    embedding_model: str
    chunk_size: int


class STTRequest(BaseModel):
    audio_data: str  # Base64 encoded
    sample_rate: Optional[int] = 16000


class VLLMTokenRequest(BaseModel):
    token: str


class VLLMModelRequest(BaseModel):
    model_name: str
    dtype: Optional[str] = "auto"
    max_model_len: Optional[int] = None
    gpu_memory_utilization: Optional[float] = 0.9


# Local embedding function
class LocalHFEmbeddingFunction:
    """Local HuggingFace embedding function with batching"""

    def __init__(self, model_name: str, device: str = None, batch_size: int = 64):
        if device is None:
            device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    def __call__(self, input):
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


# Configuration utilities
def load_rag_config() -> Dict:
    try:
        if CONFIG_PATH.exists():
            return json.loads(CONFIG_PATH.read_text())
    except Exception:
        pass
    return dict(DEFAULT_RAG_CONFIG)


def save_rag_config_local(cfg: Dict) -> None:
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    except Exception as e:
        logger.warning(f"Failed to persist RAG config: {e}")


# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown"""
    app.state.start_time = time.time()
    logger.info("LLM WebUI API starting up...")
    
    # Log environment info
    config_info = Config.get_dynamic_config()
    logger.info(f"Environment: {'containerized' if config_info['is_containerized'] else 'native'}")
    logger.info(f"Upload folder: {UPLOAD_FOLDER}")
    logger.info(f"Storage path: {config_info['storage_dir']}")

    # Clear uploads on startup
    try:
        if UPLOAD_FOLDER.exists():
            shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
        UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        if config_info["is_containerized"]:
            os.chmod(UPLOAD_FOLDER, 0o755)
        logger.info("Uploads folder cleared")
    except Exception as e:
        logger.warning(f"Failed to clear uploads: {e}")

    # Initialize services
    global rag_service
    rag_service = RAGService()
    logger.info("RAG service ready")

    # Restore configuration
    try:
        cfg = load_rag_config()
        provider = str(cfg.get("provider", "huggingface")).lower()
        model_name = str(cfg.get("embedding_model", DEFAULT_RAG_CONFIG["embedding_model"]))
        chunk_size = int(cfg.get("chunk_size", DEFAULT_RAG_CONFIG["chunk_size"]))

        if provider == "ollama":
            rag_service.embedding_function = OllamaEmbeddingFunction(
                model_name=model_name,
                url=f"{config_info['ollama_url']}/api/embeddings",
            )
        else:
            rag_service.embedding_function = LocalHFEmbeddingFunction(model_name=model_name)
        
        rag_service.default_chunk_size = chunk_size
        logger.info(f"RAG config restored: provider={provider}, model={model_name}")
    except Exception as e:
        logger.warning(f"Failed to restore RAG config: {e}")

    # Performance tuning
    try:
        cpu_threads = max(1, (os.cpu_count() or 4) - 1)
        os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
        if torch:
            torch.set_num_threads(cpu_threads)
        logger.info(f"Thread optimization applied: {cpu_threads} threads")
    except Exception as e:
        logger.warning(f"Thread tuning failed: {e}")

    logger.info("Services initialized successfully")

    yield

    # Cleanup on shutdown
    try:
        if rag_service:
            await rag_service.reset_system()
        if UPLOAD_FOLDER.exists():
            shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
        logger.info("Cleanup completed")
    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")

    logger.info("LLM WebUI API shutting down...")


# FastAPI app
app = FastAPI(
    title="LLM WebUI API",
    description="Production API for LLM WebUI with Chat and RAG capabilities",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates
try:
    templates = Jinja2Templates(directory="app/templates")
except Exception:
    templates = None


# Health endpoints
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "LLM WebUI API"}


@app.get("/health/ready")
async def readiness_check():
    """Container readiness check"""
    try:
        status = {
            "status": "ready",
            "timestamp": time.time(),
            "checks": {}
        }
        
        # Check RAG service
        if rag_service is None:
            status["checks"]["rag_service"] = {"status": "failed", "error": "Not initialized"}
            status["status"] = "not_ready"
        else:
            status["checks"]["rag_service"] = {"status": "ready"}
        
        # Check directories
        critical_dirs = [UPLOAD_FOLDER]
        for dir_path in critical_dirs:
            if dir_path.exists() and os.access(dir_path, os.R_OK | os.W_OK):
                status["checks"][dir_path.name] = {"status": "ready", "path": str(dir_path)}
            else:
                status["checks"][dir_path.name] = {"status": "failed", "path": str(dir_path)}
                status["status"] = "not_ready"
        
        return status if status["status"] == "ready" else JSONResponse(status_code=503, content=status)
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(status_code=503, content={"status": "error", "error": str(e)})


# Model endpoints
@app.get("/models")
async def get_available_models():
    """Get available Ollama models"""
    try:
        try:
            from app.services.host_service_manager import host_service_manager
            ollama_url = host_service_manager.environment_config.ollama_url
        except ImportError:
            ollama_url = Config.OLLAMA_API_URL
        
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{ollama_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            return {"models": [model["name"] for model in data.get("models", [])]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not fetch models: {str(e)}")


# Chat endpoints
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequestEnhanced):
    """Single chat message processing"""
    logger.info(f"Chat request - Backend: {request.backend}, Model: {request.model}")
    
    try:
        if request.backend == "vllm":
            if request.hf_token:
                token_result = vllm_service.set_hf_token(request.hf_token)
                if token_result["status"] == "error":
                    raise HTTPException(status_code=400, detail=token_result["message"])
            
            if not await vllm_service.is_server_running():
                raise HTTPException(status_code=400, detail="vLLM server is not running")
            
            messages = [{"role": "user", "content": request.message}]
            response_text = ""
            async for chunk in vllm_service.stream_vllm_chat(
                messages=messages, model=request.model, temperature=request.temperature
            ):
                response_text += chunk
            
            return ChatResponse(response=response_text, model=request.model)
        else:
            response = await process_llm_request(request.message, request.model, request.temperature)
            return ChatResponse(response=response, model=request.model)
            
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequestEnhanced):
    """Stream chat responses"""
    logger.info(f"Streaming chat - Backend: {request.backend}, Model: {request.model}")
    
    async def event_generator():
        try:
            if request.backend == "vllm":
                if request.hf_token:
                    token_result = vllm_service.set_hf_token(request.hf_token)
                    if token_result["status"] == "error":
                        yield f"Error: {token_result['message']}\n"
                        return
                
                if not await vllm_service.is_server_running():
                    yield "Error: vLLM server is not running\n"
                    return
                
                messages = [{"role": "user", "content": request.message}]
                async for chunk in vllm_service.stream_vllm_chat(
                    messages=messages, model=request.model, temperature=request.temperature
                ):
                    yield chunk + "\n"
            else:
                messages = [{"role": "user", "content": request.message}]
                async for chunk in stream_ollama(messages, model=request.model, temperature=request.temperature):
                    yield chunk + "\n"
                    
        except Exception as e:
            logger.error(f"Chat streaming error: {e}")
            yield f"Error: {str(e)}\n"

    return StreamingResponse(event_generator(), media_type="text/plain")


# RAG endpoints
@app.get("/api/rag/status")
async def get_rag_status():
    """Get RAG system status"""
    try:
        if not rag_service:
            return {"status": "not_initialized", "documents": 0, "chunks": 0}

        uploaded_files = len([f for f in UPLOAD_FOLDER.iterdir() if f.is_file()])
        
        collection_count = 0
        try:
            if getattr(rag_service, "text_collection", None):
                collection_count = rag_service.text_collection.count()
        except Exception:
            pass

        return {
            "status": "ready",
            "uploaded_files": uploaded_files,
            "indexed_chunks": collection_count,
            "bm25_corpus_size": len(rag_service.bm25_corpus) if rag_service.bm25_corpus else 0,
        }
    except Exception as e:
        logger.error(f"RAG status error: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/api/rag/save_config")
async def save_rag_config(request: RAGConfigRequest):
    """Save RAG configuration"""
    try:
        if not rag_service:
            raise HTTPException(status_code=503, detail="RAG service not initialized")

        # Save configuration
        cfg = load_rag_config()
        cfg.update({
            "provider": request.provider,
            "embedding_model": request.embedding_model,
            "chunk_size": int(request.chunk_size)
        })
        save_rag_config_local(cfg)

        # Update embedding function
        if request.provider.lower() == "ollama":
            current_config = Config.get_dynamic_config()
            rag_service.embedding_function = OllamaEmbeddingFunction(
                model_name=request.embedding_model,
                url=f"{current_config['ollama_url']}/api/embeddings",
            )
        elif request.provider.lower() == "huggingface":
            rag_service.embedding_function = LocalHFEmbeddingFunction(model_name=request.embedding_model)
        else:
            raise HTTPException(status_code=400, detail="Invalid provider")

        rag_service.default_chunk_size = int(request.chunk_size)
        
        return {"message": "Configuration saved successfully", "status": "success"}

    except Exception as e:
        logger.error(f"Save config error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save configuration: {str(e)}")


@app.post("/api/rag/upload")
async def rag_upload(files: List[UploadFile] = File(...), chunk_size: int = Form(500)):
    """Upload and process files for RAG"""
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    results = []
    saved_paths = []
    
    # Save files and check for duplicates
    for up in files:
        # Save file first
        dest = UPLOAD_FOLDER / up.filename
        with open(dest, "wb") as f:
            f.write(await up.read())
        
        # Check for duplicates using the saved file path
        is_duplicate, reason = rag_service._is_file_already_processed(str(dest), up.filename)
        
        if is_duplicate:
            # Remove the saved file since it's a duplicate
            dest.unlink(missing_ok=True)
            results.append({
                "filename": up.filename,
                "status": "duplicate",
                "message": reason,
                "chunks_created": 0
            })
            continue
        
        saved_paths.append((dest, up.filename))

    # Process files
    async def process_files():
        sem = asyncio.Semaphore(2)

        async def process_one(path_info):
            async with sem:
                path, filename = path_info
                try:
                    result = await rag_service.add_document(str(path), filename, chunk_size=chunk_size)
                    results.append({
                        "filename": filename,
                        **(result or {"status": "success", "message": f"Processed {filename}", "chunks_created": 0})
                    })
                except Exception as e:
                    results.append({
                        "filename": filename,
                        "status": "error",
                        "message": str(e),
                        "chunks_created": 0
                    })

        if saved_paths:
            await asyncio.gather(*(process_one(path_info) for path_info in saved_paths))

    await process_files()

    # Summary
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


@app.post("/api/rag/check-duplicates")
async def check_file_duplicates(files: List[UploadFile] = File(...)):
    """Check if uploaded files are duplicates without processing them."""
    try:
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


@app.post("/api/rag/reset")
async def reset_rag_system():
    """Reset RAG system"""
    try:
        if not rag_service:
            raise HTTPException(status_code=503, detail="RAG service not initialized")

        # Clear files
        files_removed = 0
        for file_path in UPLOAD_FOLDER.iterdir():
            if file_path.is_file():
                file_path.unlink()
                files_removed += 1

        # Reset service
        await rag_service.reset_system()
        
        return {
            "status": "success",
            "message": f"RAG system reset. Removed {files_removed} files.",
            "files_removed": files_removed
        }
    except Exception as e:
        logger.error(f"Reset error: {e}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/api/rag/query")
async def rag_query(request: RAGQueryRequestEnhanced):
    """Query RAG system"""
    try:
        if request.backend == "vllm":
            if request.hf_token:
                vllm_service.set_hf_token(request.hf_token)

            # Get context from RAG
            context = ""
            try:
                if request.use_hybrid_search and getattr(rag_service, "bm25_model", None):
                    documents = rag_service.hybrid_search(request.query, request.n_results)
                    context = " ".join(documents)
                else:
                    tc = getattr(rag_service, "text_collection", None)
                    if tc:
                        results = tc.query(query_texts=[request.query], n_results=request.n_results)
                        if results and results.get('documents'):
                            context = " ".join(results['documents'][0])
            except Exception as e:
                logger.warning(f"Vector query failed: {e}")

            # Create messages
            messages = request.messages + [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": f"Context: {context}\n\nQuery: {request.query}"}
            ]

            # Stream response
            async def event_generator():
                async for chunk in vllm_service.stream_vllm_chat(messages=messages, model=request.model):
                    yield chunk + "\n"

            return StreamingResponse(event_generator(), media_type="text/plain")
        else:
            # Ollama RAG
            async def event_generator():
                async for chunk in rag_service.query(
                    request.query, request.system_prompt, request.messages,
                    request.n_results, request.use_hybrid_search, request.model
                ):
                    yield chunk + "\n"

            return StreamingResponse(event_generator(), media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Speech-to-Text endpoints
@app.post("/api/stt/transcribe")
async def transcribe_audio(request: STTRequest):
    """Transcribe audio data"""
    try:
        import base64
        
        audio_bytes = base64.b64decode(request.audio_data)
        speech_service = get_speech_service()
        transcription = speech_service.transcribe_audio(audio_bytes, request.sample_rate)
        
        return {
            "status": "success" if transcription else "error",
            "transcription": transcription or "",
            "length": len(transcription) if transcription else 0
        }
    except Exception as e:
        logger.error(f"STT error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


# vLLM endpoints
@app.post("/api/vllm/set-token")
async def set_vllm_token(request: VLLMTokenRequest):
    """Set HuggingFace token"""
    result = vllm_service.set_hf_token(request.token)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.post("/api/vllm/start-server")
async def start_vllm_server(request: VLLMModelRequest):
    """Start vLLM server"""
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
    """Stop vLLM server"""
    return await vllm_service.stop_vllm_server()


@app.get("/api/vllm/server-status")
async def vllm_server_status():
    """Check vLLM server status"""
    return {
        "running": await vllm_service.is_server_running(),
        "current_model": vllm_service.current_model
    }


@app.get("/api/vllm/models")
async def vllm_models():
    """Return available vLLM models. Popular models removed per UI request."""
    try:
        data = vllm_service.get_available_models()
        return {"local_models": data.get("local_models", []), "popular_models": []}
    except Exception as e:
        logger.error(f"Failed to list vLLM models: {e}")
        return {"local_models": [], "popular_models": []}


@app.post("/api/vllm/download-model")
async def vllm_download_model(request: VLLMModelRequest):
    """Download a vLLM model into the local Hugging Face cache."""
    try:
        result = await vllm_service.download_model(request.model_name)
        if result.get("status") != "success":
            raise HTTPException(status_code=400, detail=result.get("message", "Download failed"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Utility functions
async def process_llm_request(message: str, model: str, temperature: float) -> str:
    """Process single LLM request"""
    messages = [{"role": "user", "content": message}]
    response = ""
    async for chunk in stream_ollama(messages, model=model, temperature=temperature):
        response += chunk
    return response


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(status_code=404, content={"error": "Endpoint not found", "path": str(request.url.path)})


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    return JSONResponse(status_code=500, content={"error": "Internal server error", "detail": str(exc)})


# Server runner
def run():
    """Run the server with graceful shutdown"""
    config = uvicorn.Config(
        "main:app", 
        host="localhost", 
        port=8000, 
        reload=bool(int(os.getenv("RELOAD", "0"))),
        log_level="info"
    )
    server = uvicorn.Server(config)
    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    def handle_exit(*_args):
        logger.info("Received exit signal")
        stop_event.set()

    # Signal handling
    signals = [signal.SIGINT]
    if hasattr(signal, "SIGTERM"):
        signals.append(signal.SIGTERM)
    
    for sig in signals:
        try:
            loop.add_signal_handler(sig, handle_exit)
        except (NotImplementedError, RuntimeError):
            signal.signal(sig, lambda s, f: stop_event.set())

    # Windows keyboard handling
    def keyboard_watcher():
        try:
            while not stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            stop_event.set()

    if sys.platform.startswith("win"):
        threading.Thread(target=keyboard_watcher, daemon=True).start()

    async def main():
        server_task = loop.create_task(server.serve())
        await stop_event.wait()
        server.should_exit = True
        await server_task

    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    finally:
        logger.info("Server stopped")


if __name__ == "__main__":
    run()
