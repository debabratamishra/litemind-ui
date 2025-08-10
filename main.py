""" Complete FastAPI backend with lifespan events and proper uvicorn configuration """
from contextlib import asynccontextmanager
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from typing import Optional, List
import uvicorn
from pathlib import Path
import httpx

from app.services.ollama import stream_ollama
from app.services.rag_service import RAGService, CrewAIRAGOrchestrator
from fastapi.responses import StreamingResponse
import json
from huggingface_hub import snapshot_download
from tqdm import tqdm
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction, SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer
try:
    import torch
except Exception:
    torch = None
import logging
import asyncio
import signal
import sys
import os
import uvicorn
import threading
from fastapi.responses import JSONResponse

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Depends, Form
from typing import List
from pathlib import Path
from fastapi.responses import StreamingResponse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Upload folder setup
UPLOAD_FOLDER = Path('./uploads')
UPLOAD_FOLDER.mkdir(exist_ok=True)

rag_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """ Lifespan event handler - replaces @app.on_event decorators """
    # Startup logic
    print("🚀 LLM WebUI API starting up...")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")

    # Clear uploads folder for fresh start
    for file in UPLOAD_FOLDER.iterdir():
        if file.is_file():
            file.unlink()
    print("🗑️ Uploads folder cleared")

    # Initialize services here
    global rag_service
    rag_service = RAGService()
    print("📚 RAG service ready")
    # ===== Performance tuning: thread counts for CPU-bound ops =====
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
                logger.info(f"Torch threads set: intra={cpu_threads}")
            except Exception as e:
                logger.warning(f"Could not set Torch threads: {e}")
        logger.info(f"OMP/MKL threads set to {cpu_threads}")
    except Exception as e:
        logger.warning(f"Thread tuning skipped: {e}")
    print("💬 Chat service ready")

    yield  # App is running

    # Shutdown logic
    print("👋 LLM WebUI API shutting down...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="LLM WebUI API",
    description="Complete API for LLM WebUI application with Chat and RAG capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware for Streamlit integration
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
except:
    templates = None  # Handle case where templates directory doesn't exist

# Pydantic models
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

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for backend detection"""
    return {"status": "healthy", "service": "LLM WebUI API"}

# Models endpoint
@app.get("/models")
async def get_available_models():
    """Fetch available models from Ollama backend."""
    try:
        async with httpx.AsyncClient() as client:
            # Ollama's default REST API endpoint for listing models
            resp = await client.get("http://localhost:11434/api/tags")
            resp.raise_for_status()
            data = resp.json()
            # Ollama returns models under "models", each has a "name"
            model_names = [model["name"] for model in data.get("models", [])]
            return {"models": model_names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not fetch models from Ollama: {str(e)}")

# ================== CHAT ROUTES ==================
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Process chat messages asynchronously"""
    try:
        response = await process_llm_request(request.message, request.model, request.temperature)
        return ChatResponse(response=response, model=request.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# New streaming chat endpoint
@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    async def event_generator():
        messages = [{"role": "user", "content": request.message}]
        async for chunk in stream_ollama(messages, model=request.model, temperature=request.temperature):
            yield chunk + "\n"  # Add newline for clean chunk separation
    return StreamingResponse(event_generator(), media_type="text/plain")

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Serve chat HTML page (if using templates)"""
    if templates:
        return templates.TemplateResponse("chat.html", {"request": request})
    return HTMLResponse("<h1>Templates not configured</h1>")

# ================== RAG ROUTES ==================
@app.get("/rag", response_class=HTMLResponse)
async def rag_page(request: Request):
    """Serve RAG HTML page"""
    if templates:
        return templates.TemplateResponse("rag.html", {"request": request})
    return HTMLResponse("<h1>Templates not configured</h1>")

# Update the RAG query endpoint

@app.post("/api/rag/query")
async def rag_query_endpoint(request: RAGQueryRequest):
    async def gen():
        try:
            if request.use_multi_agent:
                orchestrator = CrewAIRAGOrchestrator(rag_service, request.model)
                async for chunk in orchestrator.query(
                    request.query, request.system_prompt, request.messages,
                    request.n_results, request.use_hybrid_search
                ):
                    yield chunk
            else:
                async for chunk in rag_service.query(
                    request.query, request.system_prompt, request.messages,
                    request.n_results, request.use_hybrid_search
                ):
                    yield chunk
        except Exception as e:
            yield f"\n[ERROR] {e}\n"
    return StreamingResponse(gen(), media_type="text/plain")

@app.get('/api/rag/documents')
async def get_uploaded_documents():
    """Get list of uploaded documents"""
    try:
        documents = [f.name for f in UPLOAD_FOLDER.iterdir() if f.is_file()]
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# NEW: Save RAG configuration endpoint
@app.post("/api/rag/save_config")
async def save_rag_config(request: RAGConfigRequest):
    """Save RAG configuration and recreate collection if needed"""
    try:
        global rag_service
        if not rag_service:
            raise HTTPException(status_code=503, detail="RAG service not initialized")
        
        # Update embedding function based on provider
        if request.provider.lower() == "ollama":
            from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
            rag_service.embedding_function = OllamaEmbeddingFunction(
                model_name=request.embedding_model,
                url="http://localhost:11434/api/embeddings"
            )
        elif request.provider.lower() == "huggingface":
            rag_service.embedding_function = LocalHFEmbeddingFunction(
                model_name=request.embedding_model,
                batch_size=64,
            )
        
        # Update chunk size
        rag_service.default_chunk_size = request.chunk_size
        
        # Recreate collection with new settings (now properly async)
        await rag_service.recreate_collection()
        
        return {"message": "RAG configuration saved successfully", "status": "success"}
        
    except Exception as e:
        logger.error(f"Error saving RAG config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save RAG configuration: {str(e)}")

# ================== UTILITY FUNCTIONS ==================
async def process_llm_request(message: str, model: str, temperature: float) -> str:
    """Process LLM request using your existing services"""
    messages = [{"role": "user", "content": message}]
    response = ""
    async for chunk in stream_ollama(messages, model=model, temperature=temperature):  # Pass temperature
        response += chunk
    return response

# ================== ERROR HANDLERS ==================
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "path": str(request.url.path)},
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


class LocalHFEmbeddingFunction:
    """Fast local embedding function with batching and device selection.
    Compatible with Chroma's embedding_function interface.
    """
    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 64):
        # Auto-select device if not provided
        if device is None:
            if torch is not None and hasattr(torch, 'cuda') and torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    def __call__(self, texts: list[str]):
        # SentenceTransformer returns numpy; convert to list of lists
        embs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embs.tolist()

# ===== EMBEDDING FUNCTIONS =====
@app.post("/api/rag/set_embedding")
async def set_embedding(request: dict):
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

                # set embedding fn
                rag_service.embedding_function = OllamaEmbeddingFunction(
                    url="http://localhost:11434/api/embeddings",
                    model_name=model_name
                )
                yield "Ollama embedding function set.\n"

            elif provider.lower() == "huggingface":
                yield f"Loading HuggingFace model: {model_name}\n"
                # Loading the model will download it on first run and cache it
                _tmp = SentenceTransformer(model_name)
                device = 'cuda' if (torch is not None and hasattr(torch, 'cuda') and torch.cuda.is_available()) else 'cpu'
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
    saved_paths = []
    for up in files:
        dest = UPLOAD_FOLDER / up.filename
        with open(dest, "wb") as f:
            f.write(await up.read())
        saved_paths.append(dest)

    # Kick off background indexing so the loop isn’t blocked
    async def _index():
        sem = asyncio.Semaphore(2)

        async def index_one(p):
            async with sem:
                await rag_service.add_document(str(p), p.name, chunk_size=int(chunk_size))

        await asyncio.gather(*(index_one(p) for p in saved_paths))
    asyncio.create_task(_index())

    return {"status": "queued", "files": [p.name for p in saved_paths]}

# main.py
@app.get("/api/processing/capabilities")
async def processing_capabilities():
    if not rag_service:
        # Service still starting up
        return {
            "status": "initializing",
            "message": "RAG service is starting up",
            "capabilities": {
                "enhanced_csv": False,
                "ocr_available": False,
                "memory_optimized": True
            }
        }
    try:
        return rag_service.get_capabilities()
    except Exception as e:
        logger.exception("Failed to get processing capabilities")
        return {
            "status": "degraded",
            "message": f"Could not determine capabilities: {str(e)}",
            "capabilities": {
                "enhanced_csv": False,
                "ocr_available": False,
                "memory_optimized": True
            }
        }


# ================== MAIN ==================
def run():
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

    def handle_exit(*args):
        print("\n⏹️ Received exit signal.")
        stop_event.set()

    # Attempt native signal handling (works on Unix, partial on Windows)
    signals = [signal.SIGINT]
    if hasattr(signal, 'SIGTERM'):
        signals.append(signal.SIGTERM)
    for sig in signals:
        try:
            loop.add_signal_handler(sig, handle_exit)
        except (NotImplementedError, RuntimeError):
            # Fallback for Windows or when not running in main thread
            signal.signal(sig, lambda s, f: stop_event.set())

    # Extra fallback for Windows: listen for KeyboardInterrupt in thread
    def keyboard_watcher():
        try:
            while not stop_event.is_set():
                pass
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
        print("\n❌ KeyboardInterrupt caught. Shutting down gracefully...")
    finally:
        print("✅ App closed.")

if __name__ == "__main__":
    run()

