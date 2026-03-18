import time

from bittensor import Keypair
from fastapi import HTTPException, Request

from app.config import ENV
from app.logger import get_logger

logger = get_logger(__name__)

TIMESTAMP_TOLERANCE = 60  # 1 minute

DEV_HOTKEY = "dev"


async def validate_hotkey_signature(request: Request) -> str:
    """
    Validate that the request is signed by a registered validator hotkey.

    Expected headers:
        X-Hotkey:    SS58 address of the validator hotkey
        X-Timestamp: Unix timestamp (seconds) as a string
        X-Signature: Hex-encoded signature of the timestamp bytes

    Returns the hotkey SS58 address on success.
    """

    hotkey = request.headers.get("X-Hotkey")
    timestamp = request.headers.get("X-Timestamp")
    signature = request.headers.get("X-Signature")

    if not all([hotkey, timestamp, signature]):
        logger.warning(
            f"Missing auth headers: path={request.url.path} "
            f"has_hotkey={bool(hotkey)} "
            f"has_timestamp={bool(timestamp)} "
            f"has_signature={bool(signature)}"
        )
        raise HTTPException(status_code=401, detail="Missing auth headers")

    # Check timestamp freshness
    try:
        ts = int(timestamp)
    except ValueError:
        logger.warning(f"Invalid timestamp format for hotkey={hotkey}")
        raise HTTPException(status_code=401, detail="Invalid timestamp format")

    if abs(time.time() - ts) > TIMESTAMP_TOLERANCE:
        logger.warning(f"Expired timestamp for hotkey={hotkey} timestamp={timestamp}")
        raise HTTPException(status_code=401, detail="Timestamp expired")

    # Verify signature
    try:
        keypair = Keypair(ss58_address=hotkey)
        if not keypair.verify(timestamp.encode(), bytes.fromhex(signature)):
            raise HTTPException(status_code=401, detail="Invalid signature")
    except Exception as e:
        logger.warning(f"Signature verification failed for hotkey={hotkey} error={e}")
        raise HTTPException(status_code=401, detail="Invalid signature")

    return hotkey


async def get_hotkey(request: Request) -> str:
    if ENV == "production":
        return await validate_hotkey_signature(request)

    return DEV_HOTKEY
