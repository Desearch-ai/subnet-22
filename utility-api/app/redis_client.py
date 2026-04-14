import os

import redis.asyncio as redis

from app.logger import get_logger

logger = get_logger(__name__)

_client: redis.Redis | None = None


async def init_redis() -> None:
    """Initialize the async Redis connection."""
    global _client
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))
    password = os.getenv("REDIS_PASSWORD") or None

    _client = redis.Redis(
        host=host,
        port=port,
        password=password,
        db=0,
        decode_responses=True,
    )
    await _client.ping()
    logger.info("Redis connected")


async def close_redis() -> None:
    """Close the Redis connection."""
    global _client
    if _client:
        await _client.close()
        _client = None
        logger.info("Redis disconnected")


def get_redis() -> redis.Redis:
    """Return the active Redis client. Raises if not initialized."""
    if _client is None:
        raise RuntimeError("Redis not initialized — call init_redis() first")
    return _client
