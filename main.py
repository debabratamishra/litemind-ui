
"""Complete FastAPI backend for LLM WebUI with chat and RAG capabilities."""
from contextlib import asynccontextmanager
import asyncio
import json
import logging
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Optional, List

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.services.ollama import stream_ollama
from app.services.rag_service import RAGService, CrewAIRAGOrchestrator
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from sentence_transformers import SentenceTransformer

try:
    import torch
except Exception:
    torch = None

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
UPLOAD_FOLDER = Path("./uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

rag_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager to handle startup and shutdown."""
    logger.info("LLM WebUI API starting up…")
    logger.info("Upload folder: %s", UPLOAD_FOLDER)

    # Clear uploads folder for fresh start
    for file in UPLOAD_FOLDER.iterdir():
        if file.is_file():
            file.unlink()
    logger.info("Uploads folder cleared")

    # Initialize services
    global rag_service
    rag_service = RAGService()
    logger.info("RAG service ready")

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
class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = "default"
    temperature: Optional[float] = 0.7


class ChatResponse(BaseModel):
    response: str
    model: str


class RAGQueryRequest(BaseModel):
    query: str
    messages: Optional[List[dict]] = []
    model: Optional[str] = "default"
    system_prompt: Optional[str] = "You are a helpful assistant."
    n_results: Optional[int] = 3
    use_multi_agent: Optional[bool] = False
    use_hybrid_search: Optional[bool] = False


class RAGConfigRequest(BaseModel):
    provider: str
    embedding_model: str
    chunk_size: int


# ---------------------------------------------------------------------------
# Health & models
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    """Lightweight health check endpoint."""
    return {"status": "healthy", "service": "LLM WebUI API"}


@app.get("/models")
async def get_available_models():
    """Return available Ollama model names from the local backend."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:11434/api/tags")
            resp.raise_for_status()
            data = resp.json()
            model_names = [model["name"] for model in data.get("models", [])]
            return {"models": model_names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not fetch models from Ollama: {str(e)}")


# ---------------------------------------------------------------------------
# Chat routes
# ---------------------------------------------------------------------------
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Process a single chat message with the selected model."""
    try:
        response = await process_llm_request(request.message, request.model, request.temperature)
        return ChatResponse(response=response, model=request.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream chat responses chunk-by-chunk for real-time updates."""

    async def event_generator():
        messages = [{"role": "user", "content": request.message}]
        async for chunk in stream_ollama(messages, model=request.model, temperature=request.temperature):
            yield chunk + "\n"

    return StreamingResponse(event_generator(), media_type="text/plain")


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


@app.post("/api/rag/query")
async def rag_query_endpoint(request: RAGQueryRequest):
    """Execute a RAG query, optionally via a multi‑agent orchestrator, and stream results."""

    async def gen():
        try:
            if request.use_multi_agent:
                orchestrator = CrewAIRAGOrchestrator(rag_service, request.model)
                async for chunk in orchestrator.query(
                    request.query,
                    request.system_prompt,
                    request.messages,
                    request.n_results,
                    request.use_hybrid_search,
                ):
                    yield chunk
            else:                
                async for chunk in rag_service.query(
                    request.query,
                    request.system_prompt,
                    request.messages,
                    request.n_results,
                    request.use_hybrid_search,
                    request.model,
                ):
                    yield chunk
        except Exception as e:
            yield f"\n[ERROR] {e}\n"

    return StreamingResponse(gen(), media_type="text/plain")


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
    """Persist RAG configuration (embedding provider/model and chunk size) and recreate the collection."""
    try:
        global rag_service
        if not rag_service:
            raise HTTPException(status_code=503, detail="RAG service not initialized")

        if request.provider.lower() == "ollama":
            rag_service.embedding_function = OllamaEmbeddingFunction(
                model_name=request.embedding_model,
                url="http://localhost:11434/api/embeddings",
            )
        elif request.provider.lower() == "huggingface":
            rag_service.embedding_function = LocalHFEmbeddingFunction(
                model_name=request.embedding_model,
                batch_size=64,
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid provider")

        rag_service.default_chunk_size = request.chunk_size
        await rag_service.recreate_collection()
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

    def __call__(self, texts: list[str]):
        """Encode a list of texts to normalized embeddings."""
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
                async with httpx.AsyncClient() as client:
                    tags = await client.get("http://localhost:11434/api/tags")
                    tags.raise_for_status()
                    models = [m["name"] for m in tags.json().get("models", [])]

                    if model_name not in models:
                        yield f"Model not found. Pulling '{model_name}' from Ollama...\n"
                        pull_url = "http://localhost:11434/api/pull"
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
                    url="http://localhost:11434/api/embeddings",
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

            yield "Recreating collection and re-indexing documents...\n"
            await rag_service.recreate_collection()
            yield "Embedding model updated successfully.\n"
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
        collection_count = rag_service.text_collection.count() if rag_service.text_collection else 0
        
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

