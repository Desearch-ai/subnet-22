from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import DB_POOL_RECYCLE_SECONDS, DB_URL

engine = create_async_engine(
    DB_URL,
    echo=False,
    pool_pre_ping=True,
    pool_recycle=DB_POOL_RECYCLE_SECONDS,
)

async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_session():
    """FastAPI dependency that yields a database session."""
    async with async_session() as session:
        yield session
