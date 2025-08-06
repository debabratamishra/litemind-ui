""" Complete FastAPI backend with lifespan events and proper uvicorn configuration """
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Depends
from fastapi.responses import RedirectResponse, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from typing import Optional, List
import uvicorn
from pathlib import Path
import os
import httpx
# Import your services
from app.services.ollama import stream_ollama
from app.services.rag_service import RAGService, CrewAIRAGOrchestrator
from fastapi.responses import StreamingResponse
import json
from huggingface_hub import snapshot_download
from tqdm import tqdm
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction, SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Upload folder setup
UPLOAD_FOLDER = Path('./uploads')
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Global variables for services (to be initialized in lifespan)
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
    print("💬 Chat service ready")

    yield  # App is running

    # Shutdown logic
    print("👋 LLM WebUI API shutting down...")
    # Cleanup resources here if needed
    if rag_service:
        # Add any cleanup logic for RAG service
        pass

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="LLM WebUI API",
    description="Complete API for LLM WebUI application with Chat and RAG capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan  # Use lifespan instead of on_event
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

@app.post('/api/chat/stream')
async def chat_stream(request: Request):
    """Streaming chat endpoint"""
    data = await request.json()
    prompt = data.get('prompt', '')
    messages = [{"role": "user", "content": prompt}]
    async def generate():
        async for chunk in stream_ollama(messages):
            yield chunk + "\n"
    return StreamingResponse(generate(), media_type="text/plain")

# ================== RAG ROUTES ==================
@app.get("/rag", response_class=HTMLResponse)
async def rag_page(request: Request):
    """Serve RAG HTML page"""
    if templates:
        return templates.TemplateResponse("rag.html", {"request": request})
    return HTMLResponse("<h1>Templates not configured</h1>")

@app.post("/api/rag/query")
async def rag_query_endpoint(request: RAGQueryRequest):
    """Process RAG query with optional multi-agent orchestration"""
    try:
        if request.use_multi_agent:
            orchestrator = CrewAIRAGOrchestrator(rag_service, model_name=request.model)
            async def event_generator():
                async for chunk in orchestrator.query(
                    user_query=request.query,
                    system_prompt=request.system_prompt,
                    messages=request.messages,
                    n_results=request.n_results
                ):
                    yield chunk
            return StreamingResponse(event_generator(), media_type="text/plain")
        else:
            async def event_generator():
                async for chunk in rag_service.query(
                    query_text=request.query,
                    system_prompt=request.system_prompt,
                    messages=request.messages,  # NEW: Pass messages to simple query
                    n_results=request.n_results
                ):
                    yield chunk
            return StreamingResponse(event_generator(), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    """Save RAG configuration and set up embedding function"""
    try:
        logger.info(f"Saving RAG config: {request.provider}, {request.embedding_model}, chunk_size: {request.chunk_size}")
        
        # Set up the embedding function based on provider
        if request.provider == "Ollama":
            rag_service.embedding_function = OllamaEmbeddingFunction(
                url="http://localhost:11434/api/embeddings",
                model_name=request.embedding_model
            )
        else:  # HuggingFace
            rag_service.embedding_function = SentenceTransformerEmbeddingFunction(
                model_name=request.embedding_model
            )
        
        # Store chunk_size for future document uploads
        rag_service.default_chunk_size = request.chunk_size
        
        # Recreate collection with new embedding function
        rag_service.recreate_collection()
        
        return {"status": "success", "message": "Configuration saved successfully"}
    except Exception as e:
        logger.error(f"Error saving RAG config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
    return {"error": "Endpoint not found", "path": str(request.url.path)}

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    return {"error": "Internal server error", "detail": str(exc)}


# ===== EMBEDDING FUNCTIONS =====
# New endpoint for setting embedding
@app.post("/api/rag/set_embedding")
async def set_embedding(request: dict):
    provider = request.get("provider")
    model_name = request.get("model_name")
    if not provider or not model_name:
        raise HTTPException(status_code=400, detail="Provider and model_name are required")

    def progress_generator():
        try:
            if provider.lower() == "ollama":
                # Check if model exists
                yield "Checking for Ollama model...\n"
                resp = httpx.get("http://localhost:11434/api/tags")
                models = [m["name"] for m in resp.json().get("models", [])]
                if model_name not in models:
                    yield f"Model not found. Pulling '{model_name}' from Ollama...\n"
                    pull_url = "http://localhost:11434/api/pull"
                    pull_payload = {"name": model_name, "stream": True}
                    with httpx.stream("POST", pull_url, json=pull_payload) as pull_resp:
                        for chunk in pull_resp.aiter_text():
                            try:
                                data = json.loads(chunk)
                                if "status" in data:
                                    yield f"{data['status']} ({data.get('completed', 0)}/{data.get('total', 0)})\n"
                            except:
                                pass
                    yield "Ollama model pulled successfully.\n"
                else:
                    yield "Ollama model already available.\n"
                
                rag_service.embedding_function = OllamaEmbeddingFunction(
                    url="http://localhost:11434/api/embeddings",  # Explicit URL for reliability
                    model_name=model_name
                )
                yield "Ollama embedding function set.\n"

            elif provider.lower() == "huggingface":
                yield f"Downloading HuggingFace model: {model_name}\n"
                # Use snapshot_download with tqdm for progress
                for path in tqdm(snapshot_download(repo_id=model_name, ignore_patterns=["*.msgpack", "*.h5"]), desc="Downloading files"):
                    yield f"Downloaded: {path}\n"
                # Load to ensure it's ready
                SentenceTransformer(model_name)
                rag_service.embedding_function = SentenceTransformerEmbeddingFunction(model_name=model_name)
                yield "HuggingFace embedding function set.\n"

            else:
                raise ValueError("Invalid provider")

            # Recreate collection and re-index
            yield "Recreating collection and re-indexing documents...\n"
            rag_service.recreate_collection()
            yield "Embedding model updated successfully.\n"
        except Exception as e:
            yield f"Error: {str(e)}\n"

    return StreamingResponse(progress_generator(), media_type="text/plain")

# In the upload endpoint, ensure files are saved and added to RAG
@app.post("/api/rag/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    failed_files = []
    try:
        for file in files:
            if not file.filename.lower().endswith(('.pdf', '.txt', '.docx')):
                failed_files.append(f"{file.filename}: Unsupported file type")
                continue
            file_path = UPLOAD_FOLDER / file.filename
            with open(file_path, "wb") as f:
                f.write(await file.read())
            try:
                await rag_service.add_document(str(file_path), file.filename)
                logger.info(f"Uploaded and indexed: {file.filename}")
            except Exception as e:
                failed_files.append(f"{file.filename}: {str(e)}")
                logger.error(f"Failed to index {file.filename}: {str(e)}")
                # Optionally remove the saved file if indexing fails: os.remove(file_path)
        
        if failed_files:
            return {"message": "Partial success - some files failed", "failed": failed_files}, 207  # 207 Multi-Status
        return {"message": "All files uploaded and indexed successfully"}
    except Exception as e:
        logger.error(f"Unexpected upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# ================== MAIN ==================
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="localhost",
        port=8000,
        reload=True,
        log_level="info"
    )
