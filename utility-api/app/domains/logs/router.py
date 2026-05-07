from datetime import datetime

from app.auth import get_hotkey
from app.db.session import get_session
from app.domains.logs.enums import QueryKind, SearchType
from app.domains.logs.models.miner_response_log import MinerResponseLog
from app.domains.logs.schemas import (
    BatchOrganicMatchResult,
    BatchOrganicSearchRequest,
    BatchOrganicSearchResponse,
    GetScoringLogsResponse,
    OrganicLogResponse,
    SaveMinerResponseLogsRequest,
    SaveMinerResponseLogsResponse,
    ScoringLogGroupResponse,
    ScoringValidatorLogResponse,
)
from app.logger import get_logger
from fastapi import APIRouter, Depends, Query
from sqlalchemy import and_, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/logs", tags=["logs"])
logger = get_logger(__name__)

SCORING_GROUP_LIMIT = 20

PAYLOAD_NETWORK_BLOCKS = ("axon", "dendrite")
REDACTED_IP = "0.0.0.0"
REDACTED_FIELDS = {
    "ip": REDACTED_IP,
    "port": None,
    "signature": None,
}


def _redact_network_block(block):
    """Mask validator/miner identity fields while keeping process_time/status."""
    if not isinstance(block, dict):
        return block
    redacted = dict(block)
    for field, replacement in REDACTED_FIELDS.items():
        if field in redacted:
            redacted[field] = replacement
    return redacted


def _strip_network_fields(payload):
    """Redact axon/dendrite identity fields from a stored payload.

    Keeps the block shape (so process_time, status_code, status_message remain
    available to clients and tests) but masks IPs, hotkeys, and signatures.
    """
    if not isinstance(payload, dict):
        return payload
    redacted = dict(payload)
    for block in PAYLOAD_NETWORK_BLOCKS:
        if block in redacted:
            redacted[block] = _redact_network_block(redacted[block])
    return redacted


def _normalize_optional_query(query: str | None) -> str | None:
    if query is None:
        return None

    normalized_query = query.strip()
    return normalized_query or None


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
                str(log.id),
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
                        response_payload=_strip_network_fields(log.response_payload),
                        reward_payload=log.reward_payload,
                    )
                    for log in sorted_logs
                ],
            )
        )

    return sorted(
        groups,
        key=lambda group: (
            -group.scoring_epoch_start.timestamp(),
            group.miner_uid is None,
            group.miner_uid if group.miner_uid is not None else 0,
            group.search_type.value,
            group.request_query,
        ),
    )


def _build_scoring_group_filter(
    scoring_epoch_start: datetime | None,
    miner_uid: int | None,
    request_query: str,
    search_type: SearchType,
):
    return and_(
        (
            MinerResponseLog.scoring_epoch_start.is_(None)
            if scoring_epoch_start is None
            else MinerResponseLog.scoring_epoch_start == scoring_epoch_start
        ),
        (
            MinerResponseLog.miner_uid.is_(None)
            if miner_uid is None
            else MinerResponseLog.miner_uid == miner_uid
        ),
        MinerResponseLog.request_query == request_query,
        MinerResponseLog.search_type == search_type,
    )


async def _load_scoring_logs(
    session: AsyncSession,
    scoring_epoch_start: datetime | None,
    search_type: SearchType,
    miner_uids: list[int] | None,
    query: str | None,
    miner_coldkey: str | None = None,
) -> list[MinerResponseLog]:
    normalized_query = _normalize_optional_query(query)
    filters = [
        MinerResponseLog.query_kind == QueryKind.SCORING,
        MinerResponseLog.search_type == search_type,
    ]

    if scoring_epoch_start is not None:
        filters.append(MinerResponseLog.scoring_epoch_start == scoring_epoch_start)

    if miner_uids:
        filters.append(MinerResponseLog.miner_uid.in_(miner_uids))

    if miner_coldkey is not None:
        filters.append(MinerResponseLog.miner_coldkey == miner_coldkey)

    if normalized_query is not None:
        filters.append(MinerResponseLog.request_query.ilike(f"%{normalized_query}%"))

    if not miner_uids and normalized_query is None:
        limited_group_keys_stmt = (
            select(
                MinerResponseLog.scoring_epoch_start,
                MinerResponseLog.miner_uid,
                MinerResponseLog.request_query,
                MinerResponseLog.search_type,
            )
            .where(*filters)
            .distinct()
            .order_by(
                MinerResponseLog.scoring_epoch_start.desc().nullslast(),
                MinerResponseLog.miner_uid,
                MinerResponseLog.search_type,
                MinerResponseLog.request_query,
            )
            .limit(SCORING_GROUP_LIMIT)
        )
        group_key_rows = (await session.execute(limited_group_keys_stmt)).all()

        if not group_key_rows:
            return []

        filters.append(
            or_(
                *[
                    _build_scoring_group_filter(
                        scoring_epoch_start=row[0],
                        miner_uid=row[1],
                        request_query=row[2],
                        search_type=row[3],
                    )
                    for row in group_key_rows
                ]
            )
        )

    stmt = (
        select(MinerResponseLog)
        .where(*filters)
        .order_by(
            MinerResponseLog.scoring_epoch_start.desc().nullslast(),
            MinerResponseLog.miner_uid,
            MinerResponseLog.search_type,
            MinerResponseLog.request_query,
            MinerResponseLog.validator_uid,
            MinerResponseLog.id,
        )
    )

    result = await session.execute(stmt)
    return result.scalars().all()


@router.get("/scoring", response_model=GetScoringLogsResponse)
async def get_scoring_logs(
    scoring_epoch_start: datetime | None = Query(
        None, description="Optional UTC scoring epoch start timestamp."
    ),
    search_type: SearchType = Query(..., description="Search type to inspect."),
    miner_uids: list[int] | None = Query(
        None,
        description="Optional miner UIDs to inspect.",
    ),
    query: str | None = Query(
        None,
        description="Optional case-insensitive substring match for request_query.",
    ),
    miner_coldkey: str | None = Query(
        None,
        description="Optional miner coldkey to filter by.",
    ),
    session: AsyncSession = Depends(get_session),
):
    logs = await _load_scoring_logs(
        session=session,
        scoring_epoch_start=scoring_epoch_start,
        search_type=search_type,
        miner_uids=miner_uids,
        query=query,
        miner_coldkey=miner_coldkey,
    )

    return GetScoringLogsResponse(groups=_build_scoring_groups(logs))


ORGANIC_LOG_LIMIT = 500


def _build_organic_log_response(log: MinerResponseLog) -> OrganicLogResponse:
    return OrganicLogResponse(
        id=log.id,
        created_at=log.created_at,
        search_type=log.search_type,
        miner_uid=log.miner_uid,
        miner_hotkey=log.miner_hotkey,
        miner_coldkey=log.miner_coldkey,
        validator_uid=log.validator_uid,
        validator_hotkey=log.validator_hotkey,
        validator_coldkey=log.validator_coldkey,
        request_query=log.request_query,
        status_code=log.status_code,
        process_time=log.process_time,
    )


@router.post("/organic/search", response_model=BatchOrganicSearchResponse)
async def batch_search_organic_logs(
    body: BatchOrganicSearchRequest,
    session: AsyncSession = Depends(get_session),
):
    """Search organic logs for multiple exact queries in a single request."""
    if not body.queries:
        return BatchOrganicSearchResponse(
            matches=[], total_matched_queries=0, total_logs=0
        )

    filters = [
        MinerResponseLog.query_kind == QueryKind.ORGANIC,
        MinerResponseLog.search_type == body.search_type,
        MinerResponseLog.created_at >= body.created_at_start,
        MinerResponseLog.created_at <= body.created_at_end,
        MinerResponseLog.request_query.in_(body.queries),
    ]

    if body.validator_hotkey is not None:
        filters.append(MinerResponseLog.validator_hotkey == body.validator_hotkey)

    if body.miner_coldkey is not None:
        filters.append(MinerResponseLog.miner_coldkey == body.miner_coldkey)

    if body.miner_hotkey is not None:
        filters.append(MinerResponseLog.miner_hotkey == body.miner_hotkey)

    stmt = (
        select(MinerResponseLog)
        .where(*filters)
        .order_by(MinerResponseLog.created_at.asc())
        .limit(ORGANIC_LOG_LIMIT)
    )

    result = await session.execute(stmt)
    logs = result.scalars().all()

    # Group results by request_query
    logs_by_query: dict[str, list[OrganicLogResponse]] = {}
    for log in logs:
        logs_by_query.setdefault(log.request_query, []).append(
            _build_organic_log_response(log)
        )

    matches = [
        BatchOrganicMatchResult(request_query=query, logs=query_logs)
        for query, query_logs in logs_by_query.items()
    ]

    return BatchOrganicSearchResponse(
        matches=matches,
        total_matched_queries=len(matches),
        total_logs=len(logs),
    )
