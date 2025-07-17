from fastapi import FastAPI, Request, Depends
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from config import Config
from starlette.middleware.sessions import SessionMiddleware
from .models.user import Base

engine = create_async_engine(Config.SQLALCHEMY_DATABASE_URI, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=Config.SECRET_KEY)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

@app.get('/')
async def index(request: Request):
    if 'username' in request.session:
        return RedirectResponse(url='/chat')
    return RedirectResponse(url='/auth/signin')

from .routes import auth, chat, rag
app.include_router(auth.router, prefix='/auth')
app.include_router(chat.router, prefix='/chat')
app.include_router(rag.router, prefix='/rag')