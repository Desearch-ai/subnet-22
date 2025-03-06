import redis
import os
from .. import __version__

REDIS_HOST = os.environ.get("REDIS_HOST") or "localhost"
REDIS_PORT = os.environ.get("REDIS_PORT") or 6379

redis_client = redis.StrictRedis(
    host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True
)


if redis_client.get("version") != __version__:
    redis_client.flushdb()

redis_client.set("version", __version__)
