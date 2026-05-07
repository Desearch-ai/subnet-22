from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import DB_URL

# Transaction-mode connection pooler handles pooling itself, so use NullPool
# client-side and disable asyncpg prepared statements (incompatible with
# transaction pooling).
engine = create_async_engine(
    DB_URL,
    echo=False,
    poolclass=NullPool,
    connect_args={"statement_cache_size": 0, "prepared_statement_cache_size": 0},
)

async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_session():
    """FastAPI dependency that yields a database session."""
    async with async_session() as session:
        yield session
