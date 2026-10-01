from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer
from multidict import CIMultiDictProxy
from yarl import URL


@dataclass
class Reply:
    status: int
    headers: CIMultiDictProxy[str]
    body: bytes

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", "replace")

    def json(self):
        return json.loads(self.body)


class HttpClient:
    def __init__(self, base: str = "", headers: dict[str, str] | None = None):
        self.base = base.rstrip("/")
        self.session = aiohttp.ClientSession(
            headers=headers, timeout=aiohttp.ClientTimeout(total=60)
        )

    async def get(self, url: str, **options) -> Reply:
        return await self.request("GET", url, **options)

    async def post(self, url: str, **options) -> Reply:
        return await self.request("POST", url, **options)

    async def put(self, url: str, **options) -> Reply:
        return await self.request("PUT", url, **options)

    async def request(self, method: str, url: str, **options) -> Reply:
        # Absolute URLs may be presigned, so they go out exactly as given.
        target = URL(url, encoded=True) if "://" in url else URL(self.base + url)
        async with self.session.request(method, target, **options) as response:
            return Reply(response.status, response.headers, await response.read())

    async def aclose(self) -> None:
        await self.session.close()

    async def __aenter__(self) -> HttpClient:
        return self

    async def __aexit__(self, *_) -> None:
        await self.aclose()


@asynccontextmanager
async def serving(handler):
    app = web.Application()
    app.router.add_route("*", "/{path:.*}", handler)
    server = TestServer(app, host="127.0.0.1")
    await server.start_server()
    try:
        yield str(server.make_url("/"))
    finally:
        await server.close()
