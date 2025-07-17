from fastapi import APIRouter, Request, Depends
from fastapi.responses import RedirectResponse, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from ..services.ollama import stream_ollama

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get('/', response_class=HTMLResponse)
async def index(request: Request):
    if 'username' not in request.session:
        return RedirectResponse(url='/auth/signin', status_code=303)
    return templates.TemplateResponse("chat.html", {"request": request})

@router.post('/api/chat')
async def chat(request: Request):
    if 'username' not in request.session:
        return {"error": "Unauthorized"}, 401
    data = await request.json()
    prompt = data.get('prompt', '')
    messages = [{"role": "user", "content": prompt}]
    async def generate():
        async for chunk in stream_ollama(messages):
            yield chunk + "\n"
    return StreamingResponse(generate(), media_type="text/plain")