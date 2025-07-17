import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key')
    SQLALCHEMY_DATABASE_URI = os.getenv('SQLALCHEMY_DATABASE_URI', 'sqlite+aiosqlite:///llm_webui.db')
    OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', 'http://localhost:11434')
    CHROMA_DB_PATH = os.getenv('CHROMA_DB_PATH', './chroma_db')