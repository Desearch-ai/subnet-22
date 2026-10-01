from __future__ import annotations

from contextlib import asynccontextmanager

from aiohttp import web
from aiohttp.test_utils import TestServer


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
