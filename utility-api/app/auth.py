import time

from bittensor import Keypair
from fastapi import HTTPException, Request

from app.config import ENV
from app.logger import get_logger

logger = get_logger(__name__)

TIMESTAMP_TOLERANCE = 60  # 1 minute

DEV_HOTKEY = "dev"

HOTKEY_WHITELIST = frozenset(
    {
        "5CkmTfQH8UbYAbohFr2m3jPm5gWJ6L3XrSqHd1ijKuzz4iZ5",
        "5FbSFsLtGYVh5UwHWYvQWQCmQdftApS6V2xzknTnjkz22MUV",
        "5HdxEhjFBPMhbp1qS9vLtUQ3ZfDpP5eLAzm9w2ZFmtk6fVWV",
        "5Dw3g8BujZQwX9ae4PFmWhWthP1PHRdwqcQJrXUf7ARn5qF2",
        "5E2LP6EnZ54m3wS8s1yPvD5c3xo71kQroBw7aUVK32TKeZ5u",
        "5CsvRJXuR955WojnGMdok1hbhffZyB4N5ocrv82f3p5A2zVp",
        "5EtUYRu7sFqs9obCrN6JSNTnAAcpk98oJ27YV2EF9gC4tvzs",
        "5HQhXWGtKUH4cuYneBdJ77PyGqTbhGfoNZRdc1dVRmBL2268",
        "5FBk8NbxWLRboCqEL2oj1KXwSmmaC1nFmBBL4AzLEN5Qjs22",
        "5HbScNssaEfioJHXjcXdpyqo1AKnYjymidGF8opcF9rTFZdT",
        "5GQqVoRAWUA2wZ3CQRkbaG6fhpbgTjWNKQg2Y16FtwvUCneq",
        "5EeCK27ZN7ePNbwXunoQewRaJmU3BUpPXu9kcBm24FShBuiD",
        "5CBB4ot3bBypHanWQkEkMknTCXeewUaGwHmuQTbQ446Rsn22",
    }
)


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

    if hotkey not in HOTKEY_WHITELIST:
        logger.warning(f"Rejected non-whitelisted hotkey={hotkey}")
        raise HTTPException(status_code=403, detail="Hotkey is not whitelisted")

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
