"""A path to the task API that doesn't run through the operator's network: miners send their signed claims,
completions and abandons here, the validator forwards each one unchanged and signs a witness record of it.

The relay checks the miner's signature before forwarding, so it never vouches for, or passes on, a request the
miner did not sign; it adds no header of its own, so the API sees the miner's request byte for byte. The witness
goes back to the miner in the `X-Witness` header and is published here once a few blocks have passed, so nobody
learns in real time which relay a miner is using.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from collections.abc import Callable

import aiohttp
from aiohttp import web

from desearch import witness
from desearch.witness import REQUEST, STORAGE, WITNESS_HEADER
from neurons.validators.witness import ChainHead, WitnessStore

AUTH_HEADERS = ("X-Hotkey", "X-Timestamp", "X-Nonce", "X-Signature")
# The task API's own limits: a signed request is 60 s fresh, a miner's body at most 64 kB.
TOLERANCE_S = 60.0
MAX_BODY = 64_000
# A completion may wait 20 s on one already running, then do its own storage reads.
FORWARD_TIMEOUT = aiohttp.ClientTimeout(total=90.0, sock_connect=10.0)
PUBLISH_DELAY_BLOCKS = 5
RATE_PER_S = 2.0
BURST = 10.0
NONCE = re.compile(r"^[0-9a-f]{32}$")
PASSED_ON = ("Content-Type", "Retry-After")

log = logging.getLogger("validator")


class Bucket:
    """A token bucket per miner, so one hotkey can't use up a relay."""

    def __init__(self, rate: float = RATE_PER_S, burst: float = BURST):
        self.rate, self.burst = rate, burst
        self.left: dict[str, tuple[float, float]] = {}

    def take(self, hotkey: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        tokens, at = self.left.get(hotkey, (self.burst, now))
        tokens = min(self.burst, tokens + (now - at) * self.rate)
        if len(self.left) > 10_000:
            self.left.clear()
        if tokens < 1:
            self.left[hotkey] = (tokens, now)
            return False
        self.left[hotkey] = (tokens - 1, now)
        return True


def refused(status: int, detail: str) -> web.Response:
    return web.json_response({"detail": detail}, status=status)


def carried_receipts(status: int | None, raw: bytes) -> int:
    """How many signed receipts the API's answer carries; each names the request and must be in its log."""
    if status != 200:
        return 0
    try:
        answer = json.loads(raw)
    except ValueError:
        return 0
    if not isinstance(answer, dict):
        return 0
    return len([r for r in answer.get("receipts") or [answer.get("receipt")] if r])


class Relay:
    def __init__(
        self,
        keypair,
        api_url: str,
        store: WitnessStore,
        head: ChainHead,
        registered: Callable[[str], bool],
        http: aiohttp.ClientSession,
        publish_delay_blocks: int = PUBLISH_DELAY_BLOCKS,
        rate: Bucket | None = None,
        timeout: float | None = None,
        journal_proofs: Callable[[float], list[dict]] | None = None,
    ):
        self.keypair = keypair
        self.api_url = api_url.rstrip("/")
        self.store = store
        self.head = head
        self.registered = registered
        self.http = http
        self.publish_delay_blocks = publish_delay_blocks
        self.rate = rate or Bucket()
        self.timeout = (
            aiohttp.ClientTimeout(total=timeout) if timeout else FORWARD_TIMEOUT
        )
        self.forwarded = 0
        # The fraud proofs this validator's check of the task API's log wrote, published beside its censorship proofs.
        self.journal_proofs = journal_proofs

    def app(self) -> web.Application:
        app = web.Application(client_max_size=MAX_BODY)
        app.router.add_post("/v1/tasks/claim", self.forward)
        app.router.add_post("/v1/tasks/{task_id}/complete", self.forward)
        app.router.add_post("/v1/tasks/{task_id}/abandon", self.forward)
        app.router.add_get("/v1/relay", self.about)
        app.router.add_get("/v1/witness/requests", self.requests)
        app.router.add_get("/v1/witness/requests/{request}", self.one_request)
        app.router.add_get("/v1/witness/storage", self.storage)
        app.router.add_get("/v1/witness/proofs", self.proofs)
        app.router.add_get("/v1/journal/proofs", self.log_proofs)
        return app

    async def about(self, _: web.Request) -> web.Response:
        return web.json_response(
            {
                "relay": self.keypair.ss58_address,
                "block": self.head.block,
                "forwarded": self.forwarded,
            }
        )

    def checked(
        self, request: web.Request, body: bytes
    ) -> tuple[dict[str, str], str] | web.Response:
        """The miner's auth headers and body digest, or why the request is not forwarded."""
        auth = {name: request.headers.get(name, "") for name in AUTH_HEADERS}
        if not all(auth.values()):
            return refused(401, "missing auth headers")
        hotkey, timestamp, nonce, signature = (auth[name] for name in AUTH_HEADERS)
        if not timestamp.isdigit() or not NONCE.match(nonce) or len(hotkey) > 64:
            return refused(401, "malformed auth headers")
        if abs(time.time() - int(timestamp)) > TOLERANCE_S:
            return refused(401, "timestamp outside tolerance")
        if not self.registered(hotkey):
            return refused(403, "hotkey is not registered on this subnet")
        digest = hashlib.sha256(body).hexdigest()
        record = {
            "method": request.method,
            "path": request.path,
            "body_sha256": digest,
            "timestamp": timestamp,
            "nonce": nonce,
            "signature": signature,
            "miner": hotkey,
        }
        if not witness.miner_signed(record):
            return refused(401, "bad signature")
        if not self.rate.take(hotkey):
            return refused(429, "too many requests through this relay")
        return auth, digest

    async def forward(self, request: web.Request) -> web.Response:
        body = await request.read()
        found = self.checked(request, body)
        if isinstance(found, web.Response):
            return found
        auth, digest = found
        hotkey, timestamp, nonce, signature = (auth[name] for name in AUTH_HEADERS)
        headers = dict(auth)
        if body:
            headers["Content-Type"] = request.headers.get(
                "Content-Type", "application/json"
            )
        block, forwarded_at, started = self.head.block, time.time(), time.monotonic()
        status, raw, passed, error = None, b"", {}, None
        try:
            async with self.http.post(
                f"{self.api_url}{request.path}",
                data=body or None,
                headers=headers,
                timeout=self.timeout,
            ) as answer:
                status, raw = answer.status, await answer.read()
                passed = {
                    name: answer.headers[name]
                    for name in PASSED_ON
                    if name in answer.headers
                }
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            error = type(exc).__name__
        record = witness.sign(
            self.keypair,
            {
                "kind": REQUEST,
                "request": witness.request_hash(
                    request.method,
                    request.path,
                    digest,
                    hotkey,
                    timestamp,
                    nonce,
                    signature,
                ),
                "miner": hotkey,
                "method": request.method,
                "path": request.path,
                "body_sha256": digest,
                "timestamp": timestamp,
                "nonce": nonce,
                "signature": signature,
                "block": block,
                "forwarded_at": forwarded_at,
                "status": status,
                "response_sha256": (
                    hashlib.sha256(raw).hexdigest() if status is not None else None
                ),
                "promised": carried_receipts(status, raw),
                "answered_ms": (
                    int((time.monotonic() - started) * 1000)
                    if status is not None
                    else None
                ),
                "error": error,
            },
        )
        self.store.add(record)
        self.forwarded += 1
        seal = {WITNESS_HEADER: witness.header_value(record)}
        if status is None:
            return web.json_response(
                {"detail": f"the task API did not answer through this relay: {error}"},
                status=502,
                headers=seal,
            )
        return web.Response(status=status, body=raw, headers={**passed, **seal})

    def published_before(self) -> int:
        """Records are public only once this many blocks have passed, so none says where a miner is right now."""
        return (self.head.block or 0) - self.publish_delay_blocks

    async def requests(self, request: web.Request) -> web.Response:
        found = self.store.records(
            REQUEST,
            before_block=self.published_before(),
            after_id=_int(request.query.get("after"), 0),
            limit=_int(request.query.get("limit"), 100),
            miner=request.query.get("miner"),
        )
        return web.json_response({"relay": self.keypair.ss58_address, "records": found})

    async def one_request(self, request: web.Request) -> web.Response:
        # The hash commits to the miner's signature, so only someone who saw the request can name it.
        return web.json_response(
            {"records": self.store.by_ref(request.match_info["request"])}
        )

    async def storage(self, request: web.Request) -> web.Response:
        found = self.store.records(
            STORAGE,
            after_id=_int(request.query.get("after"), 0),
            limit=_int(request.query.get("limit"), 100),
            miner=request.query.get("miner"),
        )
        return web.json_response(
            {"witness": self.keypair.ss58_address, "records": found}
        )

    async def proofs(self, request: web.Request) -> web.Response:
        since = float(request.query.get("since", 0) or 0)
        return web.json_response(
            {"witness": self.keypair.ss58_address, "proofs": self.store.proofs(since)}
        )

    async def log_proofs(self, request: web.Request) -> web.Response:
        since = float(request.query.get("since", 0) or 0)
        found = self.journal_proofs(since) if self.journal_proofs else []
        return web.json_response(
            {"validator": self.keypair.ss58_address, "proofs": found}
        )


def _int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


async def serve(relay: Relay, host: str, port: int) -> web.AppRunner:
    runner = web.AppRunner(relay.app(), access_log=None)
    await runner.setup()
    await web.TCPSite(runner, host, port).start()
    log.info(f"Relaying miners' requests to {relay.api_url} on {host}:{port}")
    return runner
