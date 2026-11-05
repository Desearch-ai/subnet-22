"""A validator's relay, and what it does for a miner the operator tries to single out.

The world here: a stand-in task API that checks signatures as the real one does and keeps a log of request hashes;
the operator's network in front of it, as the route a miner reaches it by directly; and two validators' relays.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from bittensor.wallets import Keypair

from desearch import witness
from desearch.client import TaskApiClient, TaskApiError
from desearch.witness import REQUEST, WITNESS_HEADER, ApiLog
from neurons.miners.paths import MultiPathClient
from neurons.validators.relay import Bucket, Relay
from neurons.validators.witness import CensorshipCheck, ChainHead, WitnessStore

MINER = Keypair.create_from_uri("//relay-test-miner")
OTHER = Keypair.create_from_uri("//relay-test-other-miner")
LOG_KEY = Keypair.create_from_uri("//relay-test-task-api")
VALIDATORS = [Keypair.create_from_uri(f"//relay-test-validator-{i}") for i in range(2)]


class Api:
    """The task API as far as a relay can tell: it checks each signature, uses each nonce once and logs a signed
    receipt naming the request for every claim and completion it takes."""

    def __init__(self):
        self.seen: list[tuple[dict, bytes]] = []
        self.logged: dict[str, dict] = {}
        self.nonces: set[str] = set()
        self.censored: set[str] = set()

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/v1/tasks/claim", self.claim)
        app.router.add_post("/v1/tasks/{task_id}/complete", self.complete)
        app.router.add_get("/v1/events", self.events)
        app.router.add_get("/v1/rounds/{round_id}/log", self.round_log)
        return app

    async def taken(self, request: web.Request) -> tuple[str, str] | web.Response:
        body = await request.read()
        names = ("X-Hotkey", "X-Timestamp", "X-Nonce", "X-Signature")
        hotkey, timestamp, nonce, signature = (
            request.headers.get(n, "") for n in names
        )
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
            return web.json_response({"detail": "bad signature"}, status=401)
        if (hotkey, nonce) in self.nonces:
            return web.json_response({"detail": "nonce already used"}, status=401)
        self.nonces.add((hotkey, nonce))
        self.seen.append(({n: request.headers[n] for n in names}, body))
        if hotkey in self.censored and request.path.endswith("/complete"):
            # The operator drops this miner's completions and answers nothing.
            await asyncio.sleep(30)
        hashed = witness.request_hash(
            request.method, request.path, digest, hotkey, timestamp, nonce, signature
        )
        return hotkey, hashed

    def receipt(self, hotkey: str, hashed: str, outcome: str, task_id: str) -> dict:
        entry = {
            "hotkey": hotkey,
            "outcome": outcome,
            "seq": len(self.logged) + 1,
            "task_id": task_id,
            "request": hashed,
        }
        body = witness.receipt_body("r1", entry)
        signature = LOG_KEY.sign(witness.canonical(body)).hex()
        self.logged[hashed] = {**entry, "receipt_sig": signature}
        return {"body": body, "signature": signature}

    async def claim(self, request: web.Request) -> web.Response:
        taken = await self.taken(request)
        if isinstance(taken, web.Response):
            return taken
        hotkey, hashed = taken
        receipt = self.receipt(hotkey, hashed, "issued", "t1")
        return web.json_response(
            {
                "tasks": [
                    {
                        "task_id": "t1",
                        "upload": {"url": "https://storage.example/presigned"},
                    }
                ],
                "receipts": [receipt],
            }
        )

    async def complete(self, request: web.Request) -> web.Response:
        taken = await self.taken(request)
        if isinstance(taken, web.Response):
            return taken
        hotkey, hashed = taken
        task_id = request.match_info["task_id"]
        self.receipt(hotkey, hashed, "completed", task_id)
        return web.json_response({"task_id": task_id, "status": "open_for_validation"})

    async def events(self, request: web.Request) -> web.Response:
        wanted = request.query.get("request")
        found = [
            {"round_id": "r1", **entry}
            for hashed, entry in self.logged.items()
            if hashed == wanted
        ]
        return web.json_response({"events": found})

    async def round_log(self, _: web.Request) -> web.Response:
        return web.json_response(
            {"round_id": "r1", "entries": list(self.logged.values())}
        )


class Network:
    """The operator's network on one route to the API: it passes requests on, or drops them all once blocked."""

    def __init__(self, api_url: str, http: aiohttp.ClientSession):
        self.api_url = api_url.rstrip("/")
        self.http = http
        self.blocked = False
        self.dropped = 0

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_route("*", "/{path:.*}", self.handle)
        return app

    async def handle(self, request: web.Request) -> web.StreamResponse:
        if self.blocked:
            self.dropped += 1
            request.transport.close()
            await asyncio.sleep(1)
        body = await request.read()
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length")
        }
        async with self.http.request(
            request.method,
            f"{self.api_url}{request.path_qs}",
            data=body or None,
            headers=headers,
        ) as answer:
            return web.Response(
                status=answer.status,
                body=await answer.read(),
                content_type=answer.content_type,
            )


@dataclass
class World:
    api: Api
    api_url: str
    blocked_route: Network
    blocked_url: str
    open_url: str
    relays: list[Relay]
    relay_urls: list[str]
    stores: list[WitnessStore]
    head: ChainHead
    http: aiohttp.ClientSession
    kept: list[dict] = field(default_factory=list)


async def serving(stack: AsyncExitStack, app: web.Application) -> str:
    server = TestServer(app, host="127.0.0.1")
    await server.start_server()
    stack.push_async_callback(server.close)
    return str(server.make_url("")).rstrip("/")


@asynccontextmanager
async def world(tmp_path, registered=lambda hotkey: True):
    async with AsyncExitStack() as stack:
        http = await stack.enter_async_context(aiohttp.ClientSession())
        api = Api()
        api_url = await serving(stack, api.app())
        blocked = Network(api_url, http)
        blocked_url = await serving(stack, blocked.app())
        open_url = await serving(stack, Network(api_url, http).app())
        head = ChainHead(lambda: asyncio.sleep(0, 100))
        head.block = 100
        stores, relays, urls = [], [], []
        for i, key in enumerate(VALIDATORS):
            store = WitnessStore(str(tmp_path / f"witness-{i}.db"))
            stack.callback(store.close)
            relay = Relay(
                key,
                api_url,
                store,
                head,
                registered,
                http,
                publish_delay_blocks=5,
                timeout=0.5,
            )
            stores.append(store)
            relays.append(relay)
            urls.append(await serving(stack, relay.app()))
        yield World(
            api,
            api_url,
            blocked,
            blocked_url,
            open_url,
            relays,
            urls,
            stores,
            head,
            http,
        )


class InOrder(random.Random):
    """Paths in the order given, so a test knows which one goes first."""

    def shuffle(self, x, random=None):
        return None


async def test_the_relay_forwards_the_miners_signed_request_byte_for_byte_and_signs_a_witness(
    tmp_path,
):
    async with world(tmp_path) as w:
        kept: list[dict] = []
        async with TaskApiClient(
            w.relay_urls[0], MINER, on_witness=kept.append
        ) as client:
            answer = await client.post("/v1/tasks/claim", {"kind": "crawl", "count": 1})
        assert answer["tasks"][0]["task_id"] == "t1"
        ((headers, body),) = w.api.seen
        assert headers["X-Hotkey"] == MINER.ss58_address and json.loads(body) == {
            "kind": "crawl",
            "count": 1,
        }

        (record,) = kept
        assert (
            witness.verify(record) and record["witness"] == VALIDATORS[0].ss58_address
        )
        assert (record["nonce"], record["signature"]) == (
            headers["X-Nonce"],
            headers["X-Signature"],
        )
        assert (record["status"], record["promised"], record["block"]) == (200, 1, 100)
        assert (
            record["request"] in w.api.logged
        ), "the API logged the same request the relay witnessed"
        assert w.stores[0].by_ref(record["request"]) == [record]


async def test_a_witness_is_listed_publicly_only_once_a_few_blocks_have_passed(
    tmp_path,
):
    async with world(tmp_path) as w:
        async with TaskApiClient(w.relay_urls[0], MINER) as client:
            await client.post("/v1/tasks/claim", {"kind": "crawl", "count": 1})
        async with w.http.get(f"{w.relay_urls[0]}/v1/witness/requests") as r:
            assert (await r.json())[
                "records"
            ] == [], "nobody learns where a miner is right now"
        w.head.block = 106
        async with w.http.get(f"{w.relay_urls[0]}/v1/witness/requests") as r:
            listed = (await r.json())["records"]
        assert len(listed) == 1 and listed[0]["miner"] == MINER.ss58_address


async def test_the_relay_passes_on_nothing_the_miner_did_not_sign(tmp_path):
    async with world(
        tmp_path, registered=lambda hotkey: hotkey != OTHER.ss58_address
    ) as w:
        client = TaskApiClient(w.relay_urls[0], MINER)
        headers = client._auth("POST", "/v1/tasks/claim", b'{"count":1}')
        async with w.http.post(
            f"{w.relay_urls[0]}/v1/tasks/claim", data=b'{"count":50}', headers=headers
        ) as r:
            assert r.status == 401, "a body other than the one signed"
        stale = {
            **client._auth("POST", "/v1/tasks/claim", b"{}"),
            "X-Timestamp": "1000",
        }
        async with w.http.post(
            f"{w.relay_urls[0]}/v1/tasks/claim", data=b"{}", headers=stale
        ) as r:
            assert r.status == 401
        async with TaskApiClient(w.relay_urls[0], OTHER) as unregistered:
            with pytest.raises(TaskApiError) as refused:
                await unregistered.post("/v1/tasks/claim", {})
        assert refused.value.status == 403
        await client.aclose()
        assert w.api.seen == [] and w.stores[0].records(REQUEST) == []


async def test_a_relay_that_gets_no_answer_says_so_and_still_signs_what_it_forwarded(
    tmp_path,
):
    async with world(tmp_path) as w:
        w.relays[0].api_url = "http://127.0.0.1:9"
        kept: list[dict] = []
        async with TaskApiClient(
            w.relay_urls[0], MINER, on_witness=kept.append
        ) as client:
            with pytest.raises(TaskApiError) as failed:
                await client.post("/v1/tasks/t1/complete", {"key": "uploads/x.parquet"})
        assert failed.value.status == 502
        (record,) = kept
        assert witness.verify(record) and record["status"] is None and record["error"]
        assert witness.expects_receipt(record)


def test_one_miner_cannot_use_up_a_relay():
    bucket = Bucket(rate=1.0, burst=2.0)
    assert [bucket.take("m", now=0.0) for _ in range(3)] == [True, True, False]
    assert bucket.take("other", now=0.0), "others are not held up"
    assert bucket.take("m", now=1.0), "and the miner gets its rate back"


async def test_a_miner_blocked_on_its_own_route_completes_through_the_relays(tmp_path):
    """Targeted blocking: the operator's network drops everything from the miner's own route. Once TLS ends
    inside the API's enclave (block D) a relay's traffic can't be told apart, so blocking it means blocking every
    miner behind it, which is a public outage, not a targeted one."""
    async with world(tmp_path) as w:
        w.blocked_route.blocked = True
        kept: list[dict] = []
        miner = MultiPathClient(
            MINER,
            w.blocked_url,
            w.relay_urls,
            timeout=2.0,
            on_witness=kept.append,
            rng=InOrder(),
        )
        other = TaskApiClient(w.open_url, OTHER)
        async with miner, other:
            claimed = await miner.post("/v1/tasks/claim", {"kind": "crawl", "count": 1})
            done = await miner.post(
                "/v1/tasks/t1/complete", {"key": "uploads/x.parquet", "rows": 3}
            )
            await other.post("/v1/tasks/claim", {"kind": "crawl", "count": 1})
        assert claimed["tasks"] and done["status"] == "open_for_validation"
        assert (
            w.blocked_route.dropped >= 1
        ), "the miner's own route was tried and dropped"
        assert [p.name for p in miner.paths if p.failures] == [
            "direct"
        ], "and rests while the relays carry on"
        assert {r["path"] for r in kept} == {"/v1/tasks/claim", "/v1/tasks/t1/complete"}
        assert all(witness.verify(r) and r["request"] in w.api.logged for r in kept)
        miners = [headers["X-Hotkey"] for headers, _ in w.api.seen]
        assert miners.count(MINER.ss58_address) == 2 and OTHER.ss58_address in miners


async def test_a_completion_the_api_itself_drops_ends_in_a_censorship_proof_anyone_can_check(
    tmp_path,
):
    """The operator drops one miner's completions inside the API. Every relay it went through signed a witness;
    once the log has had its blocks and still lacks them, each witness becomes a proof.
    """
    async with world(tmp_path) as w:
        w.api.censored.add(MINER.ss58_address)
        kept: list[dict] = []
        miner = MultiPathClient(
            MINER,
            None,
            w.relay_urls,
            timeout=2.0,
            on_witness=kept.append,
            rng=InOrder(),
        )
        async with miner:
            await miner.post("/v1/tasks/claim", {"kind": "crawl", "count": 1})
            with pytest.raises(TaskApiError):
                await miner.post(
                    "/v1/tasks/t1/complete", {"key": "uploads/x.parquet", "rows": 3}
                )
            async with TaskApiClient(w.relay_urls[1], OTHER) as other:
                await other.post(
                    "/v1/tasks/t2/complete", {"key": "uploads/y.parquet", "rows": 3}
                )
        dropped = [r for r in kept if r["path"].endswith("/complete")]
        assert len(dropped) == 2 and all(
            r["status"] is None for r in dropped
        ), "both relays forwarded it, no answer"

        signer = LOG_KEY.ss58_address
        proofs = []
        for store in w.stores:
            check = CensorshipCheck(
                store,
                w.head,
                ApiLog(w.http, w.api_url, signer),
                after_blocks=50,
                source=w.api_url,
            )
            assert await check.check() == [], "not before the log has had its blocks"
        w.head.block = 150
        for store in w.stores:
            check = CensorshipCheck(
                store,
                w.head,
                ApiLog(w.http, w.api_url, signer),
                after_blocks=50,
                source=w.api_url,
            )
            proofs += await check.check()
        assert {p["request"] for p in proofs} == {r["request"] for r in dropped}
        assert all(witness.check_proof(p, after_blocks=50) for p in proofs)
        assert {p["witness"]["witness"] for p in proofs} == {
            v.ss58_address for v in VALIDATORS
        }, "two independent hands"
        assert all(
            p["miner"] == MINER.ss58_address for p in proofs
        ), "the other miner's completion was logged"
        async with w.http.get(f"{w.relay_urls[0]}/v1/witness/proofs") as r:
            assert len((await r.json())["proofs"]) == 1, "each relay publishes its own"


async def test_the_interim_log_counts_a_request_only_through_a_receipt_signed_by_the_pinned_key(
    tmp_path,
):
    async with world(tmp_path) as w:
        async with TaskApiClient(w.relay_urls[0], MINER) as client:
            await client.post("/v1/tasks/claim", {"kind": "crawl", "count": 1})
        (logged,) = w.api.logged
        assert (
            await ApiLog(w.http, w.api_url, LOG_KEY.ss58_address).included(logged)
            is True
        )
        assert (
            await ApiLog(w.http, w.api_url, LOG_KEY.ss58_address).included("0" * 64)
            is False
        )
        impostor = Keypair.create_from_uri("//someone-else").ss58_address
        assert (
            await ApiLog(w.http, w.api_url, impostor).included(logged) is None
        ), "a receipt the pinned key did not sign"


async def test_a_relay_publishes_the_fraud_proofs_its_validator_found_in_the_log(
    tmp_path,
):
    async with world(tmp_path) as w:
        proofs = [{"problem": {"what": "the shares", "severity": "fraud", "seq": 2}}]
        w.relays[0].journal_proofs = lambda since: [p for p in proofs if since <= 1.0]
        async with w.http.get(f"{w.relay_urls[0]}/v1/journal/proofs") as r:
            found = await r.json()
        assert (
            found["proofs"] == proofs
            and found["validator"] == VALIDATORS[0].ss58_address
        )
        async with w.http.get(f"{w.relay_urls[1]}/v1/journal/proofs") as r:
            assert (await r.json())[
                "proofs"
            ] == [], "a relay with no follower publishes none"
