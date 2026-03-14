from app.auth import get_hotkey
from app.db.session import get_session
from app.domains.logs.models.miner_response_log import MinerResponseLog
from app.domains.logs.schemas import (
    SaveMinerResponseLogsRequest,
    SaveMinerResponseLogsResponse,
)
from fastapi import APIRouter, Depends
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/logs", tags=["logs"])


@router.post("", response_model=SaveMinerResponseLogsResponse)
async def save_logs(
    body: SaveMinerResponseLogsRequest,
    _: str = Depends(get_hotkey),
    session: AsyncSession = Depends(get_session),
):
    if not body.logs:
        return SaveMinerResponseLogsResponse(inserted=0)

    values = [log.model_dump(mode="json") for log in body.logs]

    stmt = insert(MinerResponseLog).values(values)

    result = await session.execute(stmt)
    await session.commit()

    return SaveMinerResponseLogsResponse(inserted=result.rowcount or 0)
