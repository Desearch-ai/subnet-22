from datetime import datetime

from app.auth import get_hotkey
from app.db.session import get_session
from app.domains.logs.enums import QueryKind
from app.domains.logs.models.miner_response_log import MinerResponseLog
from app.domains.logs.schemas import (
    GetScoringLogsResponse,
    SaveMinerResponseLogsRequest,
    SaveMinerResponseLogsResponse,
    ScoringLogGroupResponse,
    ScoringValidatorLogResponse,
)
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
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


def _build_reward_stats(
    logs: list[MinerResponseLog],
) -> tuple[float | None, float | None, float | None]:
    rewards = [log.total_reward for log in logs if log.total_reward is not None]
    if not rewards:
        return None, None, None

    return min(rewards), max(rewards), sum(rewards) / len(rewards)


def _build_scoring_groups(
    logs: list[MinerResponseLog],
) -> list[ScoringLogGroupResponse]:
    grouped_logs: dict[tuple, list[MinerResponseLog]] = {}

    for log in logs:
        group_key = (
            log.scoring_epoch_start,
            log.miner_uid,
            log.request_query,
            log.search_type,
        )
        grouped_logs.setdefault(group_key, []).append(log)

    groups: list[ScoringLogGroupResponse] = []

    for group_key, group_logs in grouped_logs.items():
        scoring_epoch_start, miner_uid, request_query, search_type = group_key
        sorted_logs = sorted(
            group_logs,
            key=lambda log: (
                log.validator_uid is None,
                log.validator_uid if log.validator_uid is not None else 0,
                log.created_at,
            ),
        )
        first_log = sorted_logs[0]
        reward_min, reward_max, reward_avg = _build_reward_stats(sorted_logs)

        groups.append(
            ScoringLogGroupResponse(
                scoring_epoch_start=scoring_epoch_start,
                miner_uid=miner_uid,
                miner_hotkey=first_log.miner_hotkey,
                miner_coldkey=first_log.miner_coldkey,
                search_type=search_type,
                request_query=request_query,
                validator_count=len(sorted_logs),
                reward_min=reward_min,
                reward_max=reward_max,
                reward_avg=reward_avg,
                logs=[
                    ScoringValidatorLogResponse(
                        id=log.id,
                        created_at=log.created_at,
                        validator_uid=log.validator_uid,
                        validator_hotkey=log.validator_hotkey,
                        validator_coldkey=log.validator_coldkey,
                        status_code=log.status_code,
                        process_time=log.process_time,
                        total_reward=log.total_reward,
                        response_payload=log.response_payload,
                        reward_payload=log.reward_payload,
                    )
                    for log in sorted_logs
                ],
            )
        )

    return sorted(
        groups,
        key=lambda group: (
            group.miner_uid is None,
            group.miner_uid if group.miner_uid is not None else 0,
            group.search_type.value,
            group.request_query,
        ),
    )


@router.get("/scoring", response_model=GetScoringLogsResponse)
async def get_scoring_logs(
    scoring_epoch_start: datetime = Query(
        ..., description="UTC scoring epoch start timestamp."
    ),
    miner_uid: int | None = Query(
        None,
        description="Optional miner UID to inspect for the selected hour.",
    ),
    session: AsyncSession = Depends(get_session),
):
    stmt = select(MinerResponseLog).where(
        MinerResponseLog.query_kind == QueryKind.SCORING
    )
    stmt = stmt.where(MinerResponseLog.scoring_epoch_start == scoring_epoch_start)

    if miner_uid is not None:
        stmt = stmt.where(MinerResponseLog.miner_uid == miner_uid)

    stmt = stmt.order_by(
        MinerResponseLog.miner_uid,
        MinerResponseLog.search_type,
        MinerResponseLog.request_query,
        MinerResponseLog.validator_uid,
        MinerResponseLog.created_at,
    )

    result = await session.execute(stmt)
    logs = result.scalars().all()

    return GetScoringLogsResponse(groups=_build_scoring_groups(logs))
