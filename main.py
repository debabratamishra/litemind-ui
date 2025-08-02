"""
Complete FastAPI backend with lifespan events and proper uvicorn configuration
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Depends
from fastapi.responses import RedirectResponse, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from typing import Optional
import uvicorn
from pathlib import Path
import os
import httpx

# Import your services
from app.services.ollama import stream_ollama
from app.services.rag_service import RAGService, CrewAIRAGOrchestrator

# Upload folder setup
UPLOAD_FOLDER = Path('./uploads')
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Global variables for services (to be initialized in lifespan)
rag_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler - replaces @app.on_event decorators
    """
    # Startup logic
    print("🚀 LLM WebUI API starting up...")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    
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
    system_prompt: Optional[str] = "You are a helpful assistant."
    chunk_size: Optional[int] = 500
    n_results: Optional[int] = 3

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

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Serve chat HTML page (if using templates)"""
    if templates:
        return templates.TemplateResponse("chat.html", {"request": request})
    return HTMLResponse("<h1>Chat Interface</h1><p>Templates not configured</p>")

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
    return HTMLResponse("<h1>RAG Interface</h1><p>Templates not configured</p>")

@app.post('/api/rag/upload')
async def upload_document(file: UploadFile = File(...)):
    """Upload and process documents for RAG"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    filename = file.filename
    file_path = UPLOAD_FOLDER / filename

    # Save file
    with file_path.open('wb') as f:
        content = await file.read()
        f.write(content)

    # Process with RAG service
    try:
        if rag_service:
            await rag_service.add_document(str(file_path), filename)
        return {"message": "File uploaded and processed successfully", "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post('/api/rag/query')
async def rag_query(request: RAGQueryRequest):
    """Query documents using RAG"""
    try:
        if not rag_service:
            raise HTTPException(status_code=503, detail="RAG service not available")
        
        orchestrator = CrewAIRAGOrchestrator(rag_service)

        async def generate():
            async for chunk in orchestrator.query(
                request.query, 
                request.system_prompt, 
                request.n_results
            ):
                yield chunk + "\n"

        return StreamingResponse(generate(), media_type="text/plain")
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

# ================== UTILITY FUNCTIONS ==================

async def process_llm_request(message: str, model: str, temperature: float) -> str:
    """Process LLM request using your existing services"""
    messages = [{"role": "user", "content": message}]
    
    response = ""
    async for chunk in stream_ollama(messages):
        response += chunk
    
    return response

# ================== ERROR HANDLERS ==================

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return {"error": "Endpoint not found", "path": str(request.url.path)}

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    return {"error": "Internal server error", "detail": str(exc)}

# ================== MAIN ==================

if __name__ == "__main__":
    # FIXED: Pass app as import string to resolve the warning
    uvicorn.run(
        "main:app",  # Import string format: "filename:app_variable"
        host="localhost", 
        port=8000,
        reload=True,
        log_level="info"
    )
