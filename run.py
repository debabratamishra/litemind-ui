import uvicorn
from app import app, engine, Base
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

if __name__ == '__main__':
    asyncio.run(init_db())
    uvicorn.run(app, host='0.0.0.0', port=5100)