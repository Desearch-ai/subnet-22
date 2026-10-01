import asyncio
import json
from dataclasses import dataclass


@dataclass
class Response:
    status_code: int
    body: bytes

    def json(self):
        return json.loads(self.body)


class Client:
    """Calls an ASGI app in process, as a caller at `client` would, without starting a server."""

    def __init__(self, app, client: tuple[str, int] = ("127.0.0.1", 50000)):
        self.app = app
        self.client = client

    def post(self, path: str, json: dict | None = None, headers: dict | None = None):
        return asyncio.run(self._call("POST", path, json, headers or {}))

    async def _call(self, method: str, path: str, payload, headers: dict) -> Response:
        body = b"" if payload is None else json.dumps(payload).encode()
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                *((k.lower().encode(), v.encode()) for k, v in headers.items()),
            ],
            "client": self.client,
            "server": ("testserver", 80),
        }
        pending = [{"type": "http.request", "body": body, "more_body": False}]
        sent: list[dict] = []

        async def receive() -> dict:
            return pending.pop(0) if pending else {"type": "http.disconnect"}

        async def send(message: dict) -> None:
            sent.append(message)

        await self.app(scope, receive, send)
        status = next(m["status"] for m in sent if m["type"] == "http.response.start")
        return Response(
            status,
            b"".join(
                m.get("body", b"") for m in sent if m["type"] == "http.response.body"
            ),
        )
