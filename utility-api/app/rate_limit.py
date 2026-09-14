import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

RATE_LIMIT = 30
RATE_WINDOW_SECONDS = 60

_hits: dict[str, deque[float]] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    forwarded = (
        request.headers.get("CF-Connecting-IP")
        or request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
    )
    return forwarded or (request.client.host if request.client else "")


async def rate_limit(request: Request):
    now = time.monotonic()
    hits = _hits[_client_ip(request)]

    while hits and now - hits[0] >= RATE_WINDOW_SECONDS:
        hits.popleft()

    if len(hits) >= RATE_LIMIT:
        retry_after = int(RATE_WINDOW_SECONDS - (now - hits[0])) + 1
        raise HTTPException(
            status_code=429,
            detail="Too many requests",
            headers={"Retry-After": str(retry_after)},
        )

    hits.append(now)
