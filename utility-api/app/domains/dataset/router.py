import time

from app.auth import validate_hotkey_signature
from app.db.session import get_session
from app.domains.dataset.epoch_cache import EpochQuestionCache
from app.domains.dataset.schemas import NextQuestionResponse
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/dataset", tags=["dataset"])

# Will be set during app startup via init_epoch_cache()
_epoch_cache: EpochQuestionCache | None = None

# Per-validator rate limiting
_last_request: dict[str, float] = {}

MIN_REQUEST_INTERVAL = 2.0  # seconds


def get_epoch_cache() -> EpochQuestionCache:
    if _epoch_cache is None:
        raise RuntimeError("Epoch cache not initialized")
    return _epoch_cache


async def init_epoch_cache(netuid: int, subtensor_network: str):
    """Call this from your FastAPI lifespan/startup."""
    global _epoch_cache
    _epoch_cache = EpochQuestionCache(
        netuid=netuid, subtensor_network=subtensor_network
    )
    await _epoch_cache.initialize()


async def close_epoch_cache():
    """Call this from your FastAPI lifespan/shutdown."""
    global _epoch_cache
    if _epoch_cache:
        await _epoch_cache.close()
        _epoch_cache = None


@router.get("/next", response_model=NextQuestionResponse)
async def get_next_question(
    # hotkey: str = Depends(validate_hotkey_signature),
    session: AsyncSession = Depends(get_session),
    cache: EpochQuestionCache = Depends(get_epoch_cache),
):
    hotkey = "test"
    """
    Return one random question with search_type and target miner UID.

    Each call returns a previously-unserved (uid, search_type) combination
    for the calling validator. All validators receive the same question for
    the same (uid, search_type) pair within an epoch, ensuring consistent
    vTrust scoring.

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

    epoch_id, uid, search_type, question = await cache.get_next_question(
        session, hotkey
    )

    return NextQuestionResponse(
        epoch_id=epoch_id,
        uid=uid,
        search_type=search_type.value,
        question=question,
    )
