from fastapi import APIRouter, Request, Depends, UploadFile, File
from fastapi.responses import RedirectResponse, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from ..services.rag_service import RAGService
import os
from pathlib import Path

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")
UPLOAD_FOLDER = Path('./uploads')
UPLOAD_FOLDER.mkdir(exist_ok=True)

@router.get('/', response_class=HTMLResponse)
async def index(request: Request):
    if 'username' not in request.session:
        return RedirectResponse(url='/auth/signin', status_code=303)
    return templates.TemplateResponse("rag.html", {"request": request})

@router.post('/api/upload')
async def upload_file(request: Request, file: UploadFile = File(...)):
    if 'username' not in request.session:
        return {"error": "Unauthorized"}, 401
    if not file.filename:
        return {"error": "No file selected"}, 400
    filename = file.filename
    file_path = UPLOAD_FOLDER / filename
    with file_path.open('wb') as f:
        f.write(await file.read())
    rag_service = RAGService()
    await rag_service.add_document(str(file_path), filename)
    return {"message": "File uploaded and processed"}

@router.post('/api/query')
async def query(request: Request):
    if 'username' not in request.session:
        return {"error": "Unauthorized"}, 401
    data = await request.json()
    query_text = data.get('query', '')
    system_prompt = data.get('system_prompt', 'You are a helpful assistant.')
    chunk_size = int(data.get('chunk_size', 500))
    n_results = int(data.get('n_results', 3))
    rag_service = RAGService()
    async def generate():
        async for chunk in rag_service.query(query_text, system_prompt, n_results):
            yield chunk + "\n"
    return StreamingResponse(generate(), media_type="text/plain")