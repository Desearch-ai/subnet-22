import asyncio
from datetime import datetime, timedelta, timezone

from app.db.session import async_session
from app.domains.logs.models.miner_response_log import MinerResponseLog
from app.logger import get_logger
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

LOG_RETENTION = timedelta(days=7)
CLEANUP_INTERVAL_SECONDS = 3600
CLEANUP_BATCH_SIZE = 10_000

logger = get_logger(__name__)


async def delete_expired_logs(session: AsyncSession) -> int:
    cutoff = datetime.now(timezone.utc) - LOG_RETENTION
    expired_ids = (
        select(MinerResponseLog.id)
        .where(MinerResponseLog.created_at < cutoff)
        .limit(CLEANUP_BATCH_SIZE)
    )

    deleted = 0
    while True:
        result = await session.execute(
            delete(MinerResponseLog).where(MinerResponseLog.id.in_(expired_ids))
        )
        await session.commit()
        deleted += result.rowcount or 0
        if (result.rowcount or 0) < CLEANUP_BATCH_SIZE:
            return deleted


async def run_log_cleanup():
    while True:
        try:
            async with async_session() as session:
                deleted = await delete_expired_logs(session)
            logger.info(f"Deleted expired miner response logs: deleted={deleted}")
        except Exception:
            logger.exception("Failed to delete expired miner response logs")
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
