import redis.asyncio as redis
import os
import bittensor as bt
from .. import __version__

REDIS_HOST = os.environ.get("REDIS_HOST") or "localhost"
REDIS_PORT = os.environ.get("REDIS_PORT") or 6379

# Create async Redis client
redis_client = redis.Redis(
    host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True
)


async def initialize_redis():
    """Initialize Redis and check version"""
    current_version = await redis_client.get("version")
    if current_version != __version__:
        await redis_client.flushdb()

    await redis_client.set("version", __version__)

    bt.logging.info(f"Redis initialized with version: {__version__}")


async def close_redis():
    """Close Redis connection"""
    await redis_client.close()
