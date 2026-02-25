import time

from app.auth import get_hotkey
from app.db.session import get_session
from app.domains.dataset.question_cache import QuestionCache
from app.domains.dataset.schemas import NextQuestionResponse
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/dataset", tags=["dataset"])

# Will be set during app startup via init_question_cache()
_question_cache: QuestionCache | None = None

# Per-validator rate limiting
_last_request: dict[str, float] = {}

MIN_REQUEST_INTERVAL = 4  # seconds


def get_question_cache() -> QuestionCache:
    if _question_cache is None:
        raise RuntimeError("Question cache not initialized")
    return _question_cache


async def init_question_cache(netuid: int, subtensor_network: str):
    """Call this from your FastAPI lifespan/startup."""
    global _question_cache
    _question_cache = QuestionCache(netuid=netuid, subtensor_network=subtensor_network)
    await _question_cache.initialize()


async def close_question_cache():
    """Call this from your FastAPI lifespan/shutdown."""
    global _question_cache
    if _question_cache:
        await _question_cache.close()
        _question_cache = None


@router.get("/next", response_model=NextQuestionResponse)
async def get_next_question(
    hotkey: str = Depends(get_hotkey),
    session: AsyncSession = Depends(get_session),
    cache: QuestionCache = Depends(get_question_cache),
):
    """
    Return one random question with search_type and target miner UID.

    Each call returns a previously-unserved (uid, search_type) combination
    for the calling validator within the current UTC hour. All validators
    receive the same question for the same (uid, search_type) pair, ensuring
    consistent vTrust scoring.

    The cache resets at each UTC hour boundary (e.g. 11:00, 12:00, ...).

    Rate limited to one request per validator every 2 seconds.

    Requires headers:
        X-Hotkey:    Validator hotkey SS58 address
        X-Timestamp: Current unix timestamp (string)
        X-Signature: Hex-encoded signature of timestamp bytes
    """

    # Rate limit per validator
    now = time.time()
    last = _last_request.get(hotkey, 0)

    if now - last < MIN_REQUEST_INTERVAL:
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests. Wait {MIN_REQUEST_INTERVAL}s between calls.",
        )

    _last_request[hotkey] = now

    time_range_start, uid, search_type, question = await cache.get_next_question(
        session, hotkey
    )

    return NextQuestionResponse(
        time_range_start=time_range_start,
        uid=uid,
        search_type=search_type.value,
        question=question,
    )
