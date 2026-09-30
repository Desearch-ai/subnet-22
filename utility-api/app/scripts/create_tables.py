"""
Creates all database tables defined in SQLAlchemy models.
Use this for initial setup or development. For production, use Alembic migrations.

Usage:
    python -m app.scripts.create_tables
"""

import asyncio

from app.db.base import Base
from app.db.session import engine
from sqlalchemy import text


async def main():
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        await conn.run_sync(Base.metadata.create_all)

    await engine.dispose()

    print("Tables created successfully.")


if __name__ == "__main__":
    asyncio.run(main())
