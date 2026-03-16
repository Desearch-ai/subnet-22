from datetime import datetime
from uuid import uuid4

from app.auth import get_hotkey
from app.db.session import get_session
from app.domains.dataset.models.question import Question
from app.domains.logs.enums import QueryKind
from app.domains.logs.models.miner_response_log import MinerResponseLog
from app.domains.logs.schemas import (
    GetScoringLogsResponse,
    SaveMinerResponseLogsRequest,
    SaveMinerResponseLogsResponse,
    ScoringLogGroupResponse,
    ScoringValidatorLogResponse,
)
from app.logger import get_logger
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/logs", tags=["logs"])
logger = get_logger(__name__)


def _normalize_question_query(query: str) -> str:
    return query.strip()


def _merge_search_types(existing_search_types, new_search_types) -> list[str]:
    merged: list[str] = []

    for search_type in existing_search_types or []:
        search_type_value = getattr(search_type, "value", search_type)
        if search_type_value not in merged:
            merged.append(search_type_value)

    for search_type in new_search_types:
        search_type_value = getattr(search_type, "value", search_type)
        if search_type_value not in merged:
            merged.append(search_type_value)

    return merged


def _build_log_values(body: SaveMinerResponseLogsRequest) -> list[dict]:
    """Keep Python-native datatypes for SQLAlchemy inserts."""
    return [_sanitize_log_value(log.model_dump(mode="python")) for log in body.logs]


def _sanitize_log_value(value):
    """Strip null bytes that Postgres cannot store in text/JSONB values."""
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, dict):
        return {
            _sanitize_log_value(key): _sanitize_log_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_log_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_log_value(item) for item in value)
    return value


async def _upsert_organic_questions(
    session: AsyncSession, body: SaveMinerResponseLogsRequest
) -> None:
    search_types_by_query: dict[str, list] = {}

    for log in body.logs:
        if log.query_kind != QueryKind.ORGANIC:
            continue

        normalized_query = _normalize_question_query(log.request_query)
        if not normalized_query:
            continue

        query_search_types = search_types_by_query.setdefault(normalized_query, [])
        if log.search_type not in query_search_types:
            query_search_types.append(log.search_type)

    if not search_types_by_query:
        logger.debug("No organic questions to upsert from log batch")
        return

    result = await session.execute(
        select(Question).where(Question.query.in_(search_types_by_query.keys()))
    )
    existing_questions = result.scalars().all()
    existing_questions_by_query: dict[str, list[Question]] = {}

    for question in existing_questions:
        existing_questions_by_query.setdefault(question.query, []).append(question)

    question_values = []
    updated_question_count = 0

    for query, search_types in search_types_by_query.items():
        matching_questions = existing_questions_by_query.get(query, [])

        if not matching_questions:
            question_values.append(
                {
                    "id": uuid4(),
                    "query": query,
                    "search_types": [search_type.value for search_type in search_types],
                    "ai_search_tools": None,
                    "source": "desearch",
                }
            )
            continue

        merged_search_types = [search_type.value for search_type in search_types]

        for question in matching_questions:
            next_search_types = _merge_search_types(
                question.search_types, merged_search_types
            )

            current_search_types = _merge_search_types(question.search_types, [])
            if next_search_types == current_search_types:
                continue

            await session.execute(
                update(Question)
                .where(Question.id == question.id)
                .values(search_types=next_search_types)
            )
            updated_question_count += 1

    if question_values:
        await session.execute(insert(Question).values(question_values))

    logger.info(
        f"Organic question sync complete: "
        f"distinct_queries={len(search_types_by_query)} "
        f"inserted={len(question_values)} "
        f"updated={updated_question_count}"
    )


@router.post("", response_model=SaveMinerResponseLogsResponse)
async def save_logs(
    body: SaveMinerResponseLogsRequest,
    requester_hotkey: str = Depends(get_hotkey),
    session: AsyncSession = Depends(get_session),
):
    if not body.logs:
        logger.info(
            f"Received empty miner response log batch: "
            f"requester_hotkey={requester_hotkey}"
        )
        return SaveMinerResponseLogsResponse(inserted=0)

    try:
        values = _build_log_values(body)
        stmt = insert(MinerResponseLog).values(values)

        result = await session.execute(stmt)
        await _upsert_organic_questions(session, body)
        await session.commit()

        inserted = result.rowcount or 0
        logger.info(
            f"Saved miner response logs: "
            f"requester_hotkey={requester_hotkey} "
            f"inserted={inserted}"
        )

        return SaveMinerResponseLogsResponse(inserted=inserted)
    except Exception as e:
        logger.exception(
            f"Failed to save miner response logs: {e}"
            f"requester_hotkey={requester_hotkey}"
        )
        raise


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
