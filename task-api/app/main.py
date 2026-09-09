from __future__ import annotations

import json
import os
import time

from bittensor import Keypair
from fastapi import Depends, FastAPI, HTTPException, Request

from . import filler, ordering, queue as queuemod, rounds
from .auth import Authenticator, Caller
from .budget import Budgets
from .roundlog import RoundLog
from .registry import registry_from_env
from .seeds import seeds_from_env

REDIS_URL = os.environ.get("TASK_API_REDIS", "redis://localhost:6379/15")
LEASE_TTL_S = int(os.environ.get("TASK_API_LEASE_TTL", rounds.LEASE_TTL_S))
POLL_LIMIT_PER_SEC = float(os.environ.get("TASK_API_POLL_RATE", "10"))


class Nonces:
    def __init__(self, redis):
        self.redis = redis

    async def claim(self, hotkey: str, nonce: str, ttl: int) -> bool:
        return bool(
            await self.redis.set(f"nonce:{hotkey}:{nonce}", "1", nx=True, ex=ttl)
        )


class State:
    def __init__(self, redis):
        self.redis = redis
        self.queue = queuemod.Queue(redis, LEASE_TTL_S)
        self.budgets = Budgets(os.environ.get("TASK_API_BUDGETS", "budgets.db"))
        self.log = RoundLog(os.environ.get("TASK_API_LOG", "roundlog.db"))
        self.registry = registry_from_env()
        self.seeds = seeds_from_env()
        self.rounds: dict[str, rounds.Round] = {}
        self.current: str | None = None
        self.key = Keypair.create_from_uri(
            os.environ.get("TASK_API_KEY_URI", "//TaskApi")
        )

    def receipt(self, body: dict) -> str:
        return self.key.sign(ordering.canonical(body)).hex()

    async def rate_limited(self, hotkey: str) -> bool:
        key = f"rl:hotkey:{hotkey}:{int(time.time())}"
        count = await self.redis.incr(key)
        if count == 1:
            await self.redis.expire(key, 2)
        return count > POLL_LIMIT_PER_SEC


def create_app(redis=None) -> FastAPI:
    import redis.asyncio as aioredis

    app = FastAPI(title="Desearch Task API", version="0.1.0")
    app.state.core = State(redis or aioredis.from_url(REDIS_URL, decode_responses=True))

    async def caller(request: Request) -> Caller:
        core = request.app.state.core
        return await Authenticator(core.registry, Nonces(core.redis))(request)

    @app.on_event("startup")
    async def _startup():
        core = app.state.core
        await core.queue.register()

        async def janitor():
            import asyncio

            while True:
                try:
                    await filler.reclaim_expired(core)
                except Exception:
                    pass
                await asyncio.sleep(1.0)

        import asyncio

        core.janitor = asyncio.create_task(janitor())

    # --- miner endpoints ---------------------------------------------------

    @app.post("/v1/tasks/lease")
    async def lease(request: Request, who: Caller = Depends(caller)):
        core = request.app.state.core
        round_id = core.current or ""

        if await core.rate_limited(who.hotkey):
            return _refused(
                core, round_id, who,
                {"code": "RATE_LIMITED", "inputs": {"per_sec": POLL_LIMIT_PER_SEC}},
                await core.queue.next_seq(),
            )

        budget = core.budgets.get(who.hotkey).budget
        try:
            got = await core.queue.lease(who.hotkey, budget)
        except queuemod.Refusal as refusal:
            return _refused(core, round_id, who, refusal.as_dict(), await core.queue.next_seq())

        body = {
            "round_id": got.payload.get("round_id", round_id),
            "task_id": got.task_id,
            "hotkey": who.hotkey,
            "requested_at": who.requested_at,
            "outcome": "issued",
        }
        signature = core.receipt(body)
        core.budgets.assign(who.hotkey, got.payload.get("url_count", 0))
        core.log.record(
            body["round_id"], who.hotkey, who.requested_at, "issued", signature,
            task_id=got.task_id, seq=got.seq,
        )
        return {
            "task": {
                "task_id": got.task_id,
                "expires_at": got.expires_at,
                "hosts": got.payload["hosts"],
                "urls": got.payload.get("urls", []),
                "crawl_delay": got.payload.get("crawl_delay", {}),
                "upload_url": got.payload.get("upload_url"),
            },
            "receipt": {"body": body, "signature": signature},
        }

    @app.post("/v1/tasks/{task_id}/complete")
    async def complete(task_id: str, request: Request, who: Caller = Depends(caller)):
        core = request.app.state.core
        if await core.queue.holder(task_id) != who.hotkey:
            raise HTTPException(409, "you do not hold this lease")
        report = await request.json()
        records = report.get("records", [])
        payload = json.loads(await core.redis.get(f"task:{task_id}") or "{}")
        assigned = {u["url"] for u in payload.get("urls", [])}
        returned = {r.get("url") for r in records} & assigned
        missing = len(assigned) - len(returned)
        core.budgets.returned(who.hotkey, len(returned))
        seq = await core.queue.release(task_id, who.hotkey)
        core.log.record(
            core.current or "", who.hotkey, who.requested_at, "completed",
            core.receipt({"task_id": task_id, "outcome": "completed"}), task_id=task_id, seq=seq,
        )
        return {
            "task_id": task_id,
            "assigned": len(assigned),
            "accepted": len(returned),
            "missing": missing,
            "verdict": "pending_validation",
        }

    @app.post("/v1/tasks/{task_id}/abandon")
    async def abandon(task_id: str, request: Request, who: Caller = Depends(caller)):
        core = request.app.state.core
        if await core.queue.holder(task_id) != who.hotkey:
            raise HTTPException(409, "you do not hold this lease")
        seq = await core.queue.requeue(task_id)
        core.log.record(
            core.current or "", who.hotkey, who.requested_at, "reclaimed",
            core.receipt({"task_id": task_id, "outcome": "reclaimed"}), task_id=task_id, seq=seq,
        )
        miner = core.budgets.penalise(who.hotkey, task_id, "abandoned")
        return {"task_id": task_id, "budget": miner.budget}

    # --- local harness only; refused when a real registry is configured ------

    def _local_only(core):
        from .registry import LocalRegistry

        if not isinstance(core.registry, LocalRegistry):
            raise HTTPException(404, "not available")

    @app.post("/v1/admin/rounds/open")
    async def admin_open(request: Request):
        core = request.app.state.core
        _local_only(core)
        body = await request.json()
        urls = [rounds.Url(**u) for u in body["urls"]]
        round_ = await filler.open_round(core, urls, body.get("batch_target", 250))
        return {"round_id": round_.round_id, "manifest_hash": round_.manifest_hash,
                "seed_block": round_.seed_block, "batches": len(round_.batches)}

    @app.post("/v1/admin/rounds/{round_id}/fill")
    async def admin_fill(round_id: str, request: Request):
        core = request.app.state.core
        _local_only(core)
        return {"filled": await filler.reveal_and_fill(core, round_id)}

    @app.post("/v1/admin/rounds/{round_id}/close")
    async def admin_close(round_id: str, request: Request):
        core = request.app.state.core
        _local_only(core)
        return {"anchor_root": await filler.close_round(core, round_id)}

    # --- public, no auth ---------------------------------------------------

    @app.get("/v1/rounds/{round_id}")
    async def round_view(round_id: str, request: Request):
        core = request.app.state.core
        found = core.rounds.get(round_id)
        if not found:
            raise HTTPException(404, "no such round")
        return found.public_view()

    @app.get("/v1/rounds/{round_id}/log")
    async def round_log(round_id: str, request: Request):
        core = request.app.state.core
        return {
            "round_id": round_id,
            "entries": core.log.entries(round_id),
            "anchor_root": core.log.anchored_root(round_id),
        }

    @app.get("/v1/miners/{hotkey}")
    async def miner_view(hotkey: str, request: Request):
        core = request.app.state.core
        miner = core.budgets.get(hotkey)
        return {
            "hotkey": hotkey,
            "budget": miner.budget,
            "urls_verified": miner.urls_verified,
            "in_flight": await core.queue.in_flight(hotkey),
            "coverage": core.budgets.coverage_report().get(hotkey, {}),
            "transitions": core.budgets.history(hotkey),
        }

    @app.get("/v1/health")
    async def health(request: Request):
        core = request.app.state.core
        return {
            "queue_depth": await core.queue.depth(),
            "current_round": core.current,
            "shares": core.budgets.shares(),
            "coverage": core.budgets.coverage_report(),
        }

    return app


def _refused(core: State, round_id: str, who: Caller, refusal: dict, seq: int = 0) -> dict:
    body = {
        "round_id": round_id,
        "hotkey": who.hotkey,
        "requested_at": who.requested_at,
        "outcome": "refused",
        "refusal": refusal,
    }
    signature = core.receipt(body)
    core.log.record(
        round_id, who.hotkey, who.requested_at, "refused", signature, refusal=refusal
    )
    return {
        "task": None,
        "refusal": refusal,
        "receipt": {"body": body, "signature": signature},
    }


app = create_app()
