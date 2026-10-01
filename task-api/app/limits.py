from __future__ import annotations

import json

BODY_BYTES = 64_000
LARGE_BODY_BYTES = 16_000_000
LARGE_BODY_PATHS = ("/v1/admin/enqueue",)
LARGE_BODY_SUFFIXES = ("/score",)


def body_limit(path: str) -> int:
    if path in LARGE_BODY_PATHS or path.endswith(LARGE_BODY_SUFFIXES):
        return LARGE_BODY_BYTES
    return BODY_BYTES


class TooLarge(Exception):
    pass


class BodyLimit:
    """Refuses a request body over its path's limit before the rest of it is read."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = body_limit(scope["path"])
        declared = dict(scope["headers"]).get(b"content-length", b"")
        if declared.isdigit() and int(declared) > limit:
            await refuse(send, limit)
            return

        received = 0
        over = False
        answered = False

        async def counted():
            nonlocal received, over
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    over = True
                    raise TooLarge
            return message

        async def answer(message):
            nonlocal answered
            if over:
                if not answered:
                    answered = True
                    await refuse(send, limit)
                return
            answered = True
            await send(message)

        try:
            await self.app(scope, counted, answer)
        except TooLarge:
            if not answered:
                await refuse(send, limit)


async def refuse(send, limit: int) -> None:
    body = json.dumps({"detail": f"the body is over {limit} bytes"}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"connection", b"close"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
