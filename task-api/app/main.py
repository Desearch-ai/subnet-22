from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import secrets
import time

from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from . import lifecycle, queues
from .auth import Authenticator, Caller
from .budget import CRAWL, EMBED, SHARE_WINDOW_H, hour_of
from .models import (
    CompleteBody,
    EmbedScore,
    Enqueue,
    ClaimBody,
    Release,
    Score,
)
from .seeds import REVEAL_AFTER_BLOCKS
from .state import Nonces, State
from .storage import PARQUET, Changed
from .validations import (
    Infeasible,
    build_embed_vote,
    build_vote,
    utc_day,
)

DEFAULT_REDIS = "redis://localhost:6379/15"
JANITOR_INTERVAL_S = 1.0
ROUNDS_INTERVAL_S = 5.0
COMPLETE_LOCK_S = 300
TASKS_PAGE = 50
MAX_TASKS_PAGE = 100
# What a refused miner should wait, so idle polling does not fill the signed log.
RETRY_AFTER_S = {
    "QUEUE_EMPTY": 10.0,
    "NO_CAPACITY": 5.0,
    "ALREADY_HELD": 10.0,
    "VALIDATION_BACKLOG": 30.0,
    "KIND_CLOSED": 3600.0,
}
DEFAULT_ORIGINS = "http://localhost:5173,http://localhost:8081,http://127.0.0.1:5173,http://127.0.0.1:8081"

log = logging.getLogger("task_api")


def create_app(redis=None) -> FastAPI:
    import redis.asyncio as aioredis

    core = State(
        redis
        or aioredis.from_url(
            os.environ.get("TASK_API_REDIS", DEFAULT_REDIS), decode_responses=True
        )
    )

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        for storage in (core.storage, core.pages):
            try:
                await storage.check()
            except Exception as exc:
                raise RuntimeError(
                    f"R2 bucket {storage.bucket} is not reachable: {type(exc).__name__}"
                ) from None
        janitor = asyncio.create_task(_janitor(core))
        yield
        janitor.cancel()

    app = FastAPI(title="Desearch Task API", version="0.3.0", lifespan=lifespan)
    app.state.core = core

    @app.middleware("http")
    async def limit_reads(request: Request, call_next):
        if request.method == "GET":
            wait = await core.read_wait(request.client.host if request.client else "")
            if wait is not None:
                return JSONResponse(
                    {"detail": "too many requests"},
                    status_code=429,
                    headers={"Retry-After": str(wait)},
                )
        return await call_next(request)

    # Browsers may only read: every write is signed by a hotkey.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            origin.strip()
            for origin in os.environ.get(
                "TASK_API_CORS_ORIGINS", DEFAULT_ORIGINS
            ).split(",")
            if origin.strip()
        ],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    async def caller(request: Request) -> Caller:
        return await Authenticator(core.registry, Nonces(core.redis), core.admins)(
            request
        )

    async def validator(who: Caller = Depends(caller)) -> Caller:
        if not who.is_validator:
            raise HTTPException(403, "only validators may do this")
        return who

    async def admin(who: Caller = Depends(caller)) -> Caller:
        if not who.is_admin:
            raise HTTPException(403, "only admins may do this")
        return who

    def embed_inputs(payload: dict) -> dict:
        return {
            "model": payload["model"],
            "texts": payload["texts"],
            "input": {
                "url": core.storage.presign_get(payload["input_key"], core.claim_ttl),
                "sha256": payload["input_sha256"],
            },
        }

    @app.post("/v1/tasks/claim")
    async def claim(body: ClaimBody | None = None, who: Caller = Depends(caller)):
        if who.is_validator or who.is_admin:
            raise HTTPException(403, "validators may not claim tasks")
        kind = (body or ClaimBody()).kind
        tasks = core.tasks[kind]
        round_id = core.current.get(kind, "")

        retry_after = await core.retry_after(who.hotkey)
        if retry_after is not None:
            # Not logged: signing and storing every excess poll would let a flooder grow the log.
            refusal = {
                "code": "RATE_LIMITED",
                "inputs": {"per_sec": core.poll_rate, "retry_after": retry_after},
            }
            return {"task": None, "refusal": refusal, "receipt": None}

        if kind == EMBED and not core.embed_tasks:
            refusal = {"code": "KIND_CLOSED", "inputs": {"kind": kind}}
            return await _refused(core, round_id, who, refusal)

        locked_until = await core.db(core.budgets.locked_until, who.hotkey, kind)
        if locked_until is not None:
            refusal = {
                "code": "LOCKED_OUT",
                "inputs": {
                    "until": round(locked_until, 3),
                    "retry_after": round(locked_until - time.time(), 3),
                },
            }
            return await _refused(core, round_id, who, refusal)

        # Stop leasing work whose upload would expire before validation.
        validating = await core.validation.oldest_age()
        publishing = await core.publish.oldest_age()
        if core.max_backlog and max(validating, publishing) > core.max_backlog:
            refusal = {
                "code": "VALIDATION_BACKLOG",
                "inputs": {
                    "validation_s": validating,
                    "publish_s": publishing,
                    "limit_s": core.max_backlog,
                },
            }
            return await _refused(core, round_id, who, refusal)

        budget = (await core.db(core.budgets.get_or_create, who.hotkey, kind)).budget
        try:
            got = await tasks.claim(who.hotkey, budget)
        except queues.Refusal as refusal:
            return await _refused(core, round_id, who, refusal.as_dict())

        task_id = got.task_id
        name = f"task={task_id}/{who.hotkey}-{got.seq}.parquet"
        upload_key = f"uploads/dt={utc_day()}/{name}"
        try:
            upload_url = core.storage.presign_put(upload_key, PARQUET, core.claim_ttl)
            inputs = embed_inputs(got.payload) if kind == EMBED else {}
        except Exception:
            log.exception("could not presign the upload for %s", task_id)
            await tasks.abandon(task_id, who.hotkey)
            raise HTTPException(503, "upload signing is unavailable") from None
        await core.redis.set(
            f"issued:{task_id}",
            json.dumps({"hotkey": who.hotkey, "key": upload_key, "name": name}),
        )

        round_id = got.payload.get("round_id", round_id)
        receipt = await core.record(
            round_id, who.hotkey, who.requested_at, "issued", got.seq, task_id=task_id
        )
        return {
            "task": {
                "task_id": task_id,
                "kind": kind,
                "round_id": round_id,
                "expires_at": got.expires_at,
                "urls": got.payload.get("urls", []),
                **inputs,
                "upload": {
                    "url": upload_url,
                    "key": upload_key,
                    "content_type": PARQUET,
                    "expires_at": got.expires_at,
                },
            },
            "receipt": receipt,
        }

    @app.post("/v1/tasks/{task_id}/complete")
    async def complete(
        task_id: str, report: CompleteBody, who: Caller = Depends(caller)
    ):
        lock = f"completing:{task_id}"
        if not await core.redis.set(lock, who.hotkey, nx=True, ex=COMPLETE_LOCK_S):
            raise HTTPException(
                503,
                "a completion for this task is already running",
                {"Retry-After": "2"},
            )
        try:
            return await verify_completion(task_id, report, who)
        finally:
            await core.redis.delete(lock)

    async def verify_completion(
        task_id: str, report: CompleteBody, who: Caller
    ) -> dict:
        if await core.claim_holder(task_id) != who.hotkey:
            raise HTTPException(409, "you do not hold this claim")
        issued = json.loads(await core.redis.get(f"issued:{task_id}") or "{}")
        if report.key != issued.get("key"):
            raise HTTPException(400, "key is not the one issued with this claim")
        try:
            found = await core.storage.stat(report.key)
        except Exception:
            log.exception("HEAD failed for %s", report.key)
            raise HTTPException(502, "object storage is unavailable, retry") from None
        if not found or not found[0]:
            raise HTTPException(422, "nothing was uploaded to the issued key")
        size, etag = found
        if size > core.max_upload:
            await lifecycle.delete_quietly(core.storage, report.key)
            raise HTTPException(
                413, f"upload is {size} bytes, the limit is {core.max_upload}"
            )

        # The PUT URL outlives this call, so work on a copy the miner can't touch.
        attempt = secrets.token_hex(4)
        frozen = f"submitted/dt={utc_day()}/{issued['name'].removesuffix('.parquet')}-{attempt}.parquet"
        try:
            frozen_etag = await core.storage.copy(report.key, frozen, etag)
        except Changed:
            raise HTTPException(
                409, "the upload changed while completing, retry"
            ) from None
        except Exception:
            log.exception("could not freeze %s", report.key)
            raise HTTPException(502, "object storage is unavailable, retry") from None

        payload = await core.payload(task_id) or {}
        kind = payload.get("kind", CRAWL)
        round_id = payload.get("round_id") or core.current.get(kind, "")
        # The sample seed is a block hash nobody knew when the upload was frozen.
        frozen_block = await core.seeds.current_block()
        completed_at = time.time()
        job = {
            "task_id": task_id,
            "kind": kind,
            "round_id": round_id,
            "miner": who.hotkey,
            "key": frozen,
            "etag": frozen_etag,
            "urls": payload.get("urls", []),
            "position": payload.get("position", 0),
            "rank": payload.get("rank", payload.get("position", 0)),
            "attempts": payload.get("attempts", 0),
            "size": size,
            "reported": report.model_dump(exclude={"key"}),
            "completed_at": completed_at,
            "frozen_block": frozen_block,
            "seed_block": frozen_block + REVEAL_AFTER_BLOCKS,
            "deadline": completed_at + core.seeds.wait_s() + core.validation_ttl,
            **{
                name: payload[name]
                for name in lifecycle.EMBED_FIELDS
                if name in payload
            },
        }
        job["manifest"] = lifecycle.signed_manifest(core, job)
        try:
            await core.storage.put_json(lifecycle.manifest_key(frozen), job["manifest"])
        except Exception:
            log.exception("could not write the manifest for %s", frozen)
            await lifecycle.delete_quietly(core.storage, frozen)
            raise HTTPException(502, "object storage is unavailable, retry") from None
        seq = await core.tasks[kind].complete(task_id, who.hotkey, job, report.key)
        if seq is None:
            await lifecycle.delete_quietly(core.storage, frozen)
            raise HTTPException(409, "you do not hold a live claim on this task")
        await lifecycle.delete_quietly(core.storage, report.key)
        await lifecycle.publish_open(core)
        await core.record(
            round_id,
            who.hotkey,
            who.requested_at,
            "completed",
            seq,
            task_id=task_id,
            block=frozen_block,
        )
        return {"task_id": task_id, "status": "open_for_validation"}

    @app.post("/v1/tasks/{task_id}/abandon")
    async def abandon(task_id: str, who: Caller = Depends(caller)):
        payload = await core.payload(task_id) or {}
        kind = payload.get("kind", CRAWL)
        seq = await core.tasks[kind].abandon(task_id, who.hotkey)
        if seq is None:
            raise HTTPException(409, "you do not hold this claim")
        round_id = payload.get("round_id") or core.current.get(kind, "")
        if kind == CRAWL:
            await core.db(
                core.budgets.record_coverage,
                who.hotkey,
                len(set(payload.get("urls", []))),
                0,
            )
        await core.record(
            round_id,
            who.hotkey,
            who.requested_at,
            "reclaimed",
            seq,
            task_id=task_id,
            cause="abandoned",
        )
        miner = await core.db(
            core.budgets.penalise, who.hotkey, task_id, "abandoned", kind
        )
        return {"task_id": task_id, "budget": miner.budget}

    @app.post("/v1/validation/{task_id}/release")
    async def release(task_id: str, body: Release, who: Caller = Depends(validator)):
        """An upload storage lost is void; nothing else is anyone's to hand back."""
        job = await core.validation.job(task_id)
        if job is None:
            raise HTTPException(409, "no such open upload")
        try:
            gone = await core.storage.stat(job["key"]) is None
        except Exception:
            raise HTTPException(502, "object storage is unavailable, retry") from None
        if not gone:
            return {"task_id": task_id, "status": "open"}
        lapsed = await core.validation.finalize(task_id)
        if lapsed is None:
            raise HTTPException(409, "the upload was finalized")
        await lifecycle.void_task(
            core, task_id, lapsed.job, who.hotkey, "upload_missing"
        )
        return {"task_id": task_id, "status": "void"}

    @app.post("/v1/validation/{task_id}/score")
    async def score(
        task_id: str, result: dict = Body(...), who: Caller = Depends(validator)
    ):
        lock = lifecycle.lock_key(task_id)
        if not await core.redis.set(
            lock, who.hotkey, nx=True, ex=lifecycle.FINALIZE_LOCK_S
        ):
            raise HTTPException(
                503,
                "a verdict for this upload is already being recorded",
                {"Retry-After": "2"},
            )
        try:
            return await record_verdict(task_id, result, who)
        finally:
            await core.redis.delete(lock)

    async def record_verdict(task_id: str, result: dict, who: Caller) -> dict:
        if await core.db(core.validations.is_excluded, who.hotkey):
            raise HTTPException(403, "this validator disagreed with too many audits")
        await core.validation.present(who.hotkey)
        job = await core.validation.job(task_id)
        if job is None:
            raise HTTPException(409, "no such open upload")

        kind = job.get("kind", CRAWL)
        try:
            checked = (EmbedScore if kind == EMBED else Score).model_validate(result)
        except ValidationError as exc:
            raise HTTPException(422, json.loads(exc.json(include_url=False))) from None
        builder = build_embed_vote if kind == EMBED else build_vote
        try:
            vote = builder(job, who.hotkey, checked.model_dump())
        except Infeasible as why:
            raise HTTPException(422, str(why)) from None
        if not await core.validation.vote(task_id, who.hotkey, vote):
            raise HTTPException(409, "this validator already voted on this upload")

        finalized = await lifecycle.finalize_task(core, task_id, job, time.time())
        if finalized is not None:
            return finalized
        return {
            "task_id": task_id,
            "verdict": "pending",
            "credited": 0,
            "miner_budget": (
                await core.db(core.budgets.get, job["miner"], kind)
            ).budget,
        }

    @app.get("/v1/shares")
    async def shares():
        return {
            "window_hours": SHARE_WINDOW_H,
            "as_of": time.time() - core.ledger_delay,
            "pools": await core.db(
                core.budgets.shares, SHARE_WINDOW_H, time.time() - core.ledger_delay
            ),
        }

    @app.post("/v1/admin/enqueue")
    async def admin_enqueue(body: Enqueue, who: Caller = Depends(admin)):
        round_ = await lifecycle.open_round(core, body.urls, body.batch_target)
        return {
            "round_id": round_.round_id,
            "batches": len(round_.batches),
            "seed_block": round_.seed_block,
            "manifest_hash": round_.manifest_hash,
        }

    @app.get("/v1/key")
    async def key():
        return {"signer": core.key.ss58_address}

    @app.get("/v1/rounds")
    async def round_list(limit: int = Query(100, ge=1, le=1000)):
        return {"rounds": await core.db(core.rounds.recent, limit)}

    @app.get("/v1/rounds/{round_id}")
    async def round_view(round_id: str):
        found = await core.db(core.rounds.get, round_id)
        if not found:
            raise HTTPException(404, "no such round")
        return {**found.public_view(), "signer": core.key.ss58_address}

    @app.get("/v1/rounds/{round_id}/log")
    async def round_log(round_id: str):
        return {
            "round_id": round_id,
            "entries": await core.db(core.log.entries, round_id),
            "anchor_root": await core.db(core.log.anchored_root, round_id),
        }

    @app.get("/v1/tasks")
    async def tasks_view(
        miner: str | None = None,
        validator: str | None = None,
        since: float = 0.0,
        before: float | None = None,
        limit: int = Query(TASKS_PAGE, ge=1, le=MAX_TASKS_PAGE),
    ):
        """Scored tasks, newest first; pass `next` back as `before` for the next page."""
        visible = time.time() - core.ledger_delay
        before = visible if before is None else min(before, visible)
        tasks = await core.db(
            core.validations.recent, miner, validator, since, before, limit
        )
        full = len(tasks) == limit
        return {"tasks": tasks, "next": tasks[-1]["scored_at"] if full else None}

    @app.get("/v1/tasks/{task_id}")
    async def task_view(task_id: str):
        view = await task_state(task_id)
        return {**view, "urls": await core.db(core.validations.urls, task_id)}

    async def task_state(task_id: str) -> dict:
        scored = await core.db(core.validations.latest, task_id)
        if scored and scored["scored_at"] > time.time() - core.ledger_delay:
            scored = None
        job = await core.validation.job(task_id)
        if job:
            status = "voting" if await core.validation.voters(task_id) else "open"
            return {
                "task_id": task_id,
                "status": status,
                "round_id": job["round_id"],
                "miner": job["miner"],
                "score": scored,
            }

        payload = await core.payload(task_id)
        if payload is not None:
            holder = await core.claim_holder(task_id)
            return {
                "task_id": task_id,
                "status": "claimed" if holder else "queued",
                "round_id": payload.get("round_id"),
                "miner": holder,
                "score": scored,
            }

        if scored is None:
            raise HTTPException(404, "no such task")
        return {
            "task_id": task_id,
            "status": scored["verdict"],
            "round_id": scored["round_id"],
            "miner": scored["miner"],
            "score": scored,
        }

    @app.get("/v1/miners/{hotkey}")
    async def miner_view(hotkey: str):
        pools = {}
        for kind, tasks in core.tasks.items():
            miner = await core.db(core.budgets.get, hotkey, kind)
            pools[kind] = {
                "budget": miner.budget,
                "verified": miner.verified,
                "in_flight": await tasks.in_flight(hotkey),
                "locked_until": await core.db(core.budgets.locked_until, hotkey, kind),
            }
        return {
            "hotkey": hotkey,
            "pools": pools,
            "coverage": (await core.db(core.budgets.coverage_report)).get(hotkey, {}),
            "verdicts": await core.db(core.validations.verdicts, hotkey),
            "transitions": await core.db(core.budgets.history, hotkey),
        }

    @app.get("/v1/miners/{hotkey}/verdicts")
    async def own_verdicts(
        hotkey: str,
        since: float = 0.0,
        limit: int = Query(TASKS_PAGE, ge=1, le=MAX_TASKS_PAGE),
        who: Caller = Depends(caller),
    ):
        """A miner's own finalized verdicts, without the public ledger's delay."""
        if who.hotkey != hotkey:
            raise HTTPException(403, "only the miner itself may read this")
        tasks = await core.db(core.validations.recent, hotkey, None, since, None, limit)
        return {"tasks": tasks}

    @app.get("/v1/ping")
    async def ping():
        return {"ok": True}

    @app.get("/v1/health")
    async def health():
        return {
            "queue_depth": {
                kind: await tasks.depth() for kind, tasks in core.tasks.items()
            },
            "validation_depth": await core.validation.depth(),
            "active_validators": sorted(await core.validation.active()),
            "oldest_validation_s": await core.validation.oldest_age(),
            "publishing": await core.publish.depth(),
            "oldest_publish_s": await core.publish.oldest_age(),
            "publish_set_aside": await core.publish.dead_count(),
            "publish_lost": await core.publish.lost_count(),
            "verdicts": await core.db(core.validations.verdicts),
            "validators": await core.db(core.validations.audit_standing),
            "current_round": core.current,
            "embed_tasks": core.embed_tasks,
            "embed_model": core.embed_model,
            "pools": await core.db(core.budgets.shares),
            "coverage": await core.db(core.budgets.coverage_report),
        }

    return app


async def _janitor(core: State) -> None:
    rounds_at, pruned_hour = 0.0, None
    while True:
        try:
            await lifecycle.reclaim_expired(core)
            await lifecycle.finalize_due(core)
            await lifecycle.publish_open(core)
            await lifecycle.return_expired_publishes(core)
            if time.monotonic() - rounds_at >= ROUNDS_INTERVAL_S:
                await lifecycle.open_embed_rounds(core)
                await lifecycle.fill_missing(core)
                await lifecycle.reveal_pending(core)
                await lifecycle.close_finished(core)
                rounds_at = time.monotonic()
            if pruned_hour != hour_of():
                await core.db(core.budgets.prune)
                await core.db(core.validations.prune_urls)
                pruned_hour = hour_of()
        except Exception:
            log.exception("janitor pass failed")
        await asyncio.sleep(JANITOR_INTERVAL_S)


async def _refused(core: State, round_id: str, who: Caller, refusal: dict) -> dict:
    refusal["inputs"].setdefault("retry_after", RETRY_AFTER_S.get(refusal["code"], 5.0))
    seq = await core.next_seq()
    receipt = await core.record(
        round_id, who.hotkey, who.requested_at, "refused", seq, refusal=refusal
    )
    return {"task": None, "refusal": refusal, "receipt": receipt}


app = create_app()
