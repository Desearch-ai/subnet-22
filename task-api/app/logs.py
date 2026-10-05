from __future__ import annotations

import asyncio
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Path, Query

from . import queues
from .budget import CRAWL, HOUR, SHARE_WINDOW_H, Budgets
from .roundlog import RoundLog
from .validations import VERDICTS, Validations

MAX_WAITING = 32
CACHE_S = 5.0
CACHE_ENTRIES = 256
PAGE = 50
MAX_PAGE = 100
MAX_WINDOW_H = 168
LIVE_ROWS = 50
BUCKET_MINUTES = (5, 15, 60)
MAX_BUCKETS = 168
ID_CHARS = 64
PATHS = (
    "/v1/overview",
    "/v1/live",
    "/v1/stats",
    "/v1/tasks",
    "/v1/votes",
    "/v1/events",
    "/v1/miners",
    "/v1/validators",
)

TASK_POINT = ("tasks", *VERDICTS, "returned", "credited")
VOTE_POINT = ("votes", *VERDICTS, "agreed", "disagreed")
MINER_TOTALS = ("tasks", *VERDICTS, "returned", "missing", "credited")
VALIDATOR_TOTALS = ("votes", *VERDICTS, "agreed", "disagreed", "decided")

Hours = Annotated[int, Query(ge=1, le=MAX_WINDOW_H)]
Limit = Annotated[int, Query(ge=1, le=MAX_PAGE)]
Id = Annotated[str | None, Query(max_length=ID_CHARS)]
PathId = Annotated[str, Path(max_length=ID_CHARS)]
Verdict = Literal["pass", "fail", "void"]
Kind = Literal["crawl", "embed"]
Outcome = Literal["issued", "completed", "refused", "reclaimed", "dropped"]


class Busy(Exception):
    pass


class Logs:
    """Log reads have a connection and a thread of their own, so they never hold up a claim."""

    def __init__(self, db: sqlite3.Connection):
        self.budgets = Budgets(db)
        self.validations = Validations(db)
        self.events = RoundLog(db)
        db.execute("PRAGMA query_only = ON")
        self.thread = ThreadPoolExecutor(1, thread_name_prefix="logs")
        self.waiting = 0
        self.cache: dict[tuple, tuple[float, object]] = {}

    async def read(self, fn, *args, **kwargs):
        if self.waiting >= MAX_WAITING:
            raise Busy
        self.waiting += 1
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self.thread, partial(fn, *args, **kwargs))
        finally:
            self.waiting -= 1

    async def cached(self, key: tuple, make):
        now = time.monotonic()
        found = self.cache.get(key)
        if found and found[0] > now:
            return found[1]
        value = await make()
        if len(self.cache) >= CACHE_ENTRIES:
            self.cache.clear()
        self.cache[key] = (now + CACHE_S, value)
        return value

    def miner_rows(self, since: float, until: float) -> list[dict]:
        totals = self.validations.miner_totals(since, until)
        budgets = {miner.hotkey: miner for miner in self.budgets.all()}
        locked = self.budgets.lockouts()
        coverage = self.budgets.coverage_report(SHARE_WINDOW_H, until)
        shares = self.budgets.shares(SHARE_WINDOW_H, until).get(CRAWL, {})
        rows = []
        for hotkey in set(totals) | set(budgets):
            budget = budgets.get(hotkey) or self.budgets.get(hotkey)
            covered = coverage.get(hotkey, {})
            rows.append(
                {
                    "hotkey": hotkey,
                    "budget": budget.budget,
                    "locked_until": locked.get(hotkey),
                    "verified": budget.verified,
                    **dict.fromkeys(MINER_TOTALS, 0),
                    "last_scored_at": None,
                    **totals.get(hotkey, {}),
                    "coverage": covered.get("coverage"),
                    "share": shares.get(hotkey, 0.0),
                }
            )
        return sorted(rows, key=lambda row: (-row["share"], -row["credited"]))

    def validator_rows(
        self, since: float, until: float, last_seen: dict[str, float]
    ) -> list[dict]:
        totals = self.validations.validator_totals(since, until)
        standing = self.validations.audit_standing()
        rows = [
            validator_row(hotkey, totals, standing, last_seen)
            for hotkey in set(totals) | set(standing) | set(last_seen)
        ]
        return sorted(rows, key=lambda row: (-row["votes"], row["hotkey"]))

    def series(
        self,
        bucket_s: int,
        buckets: int,
        until: float,
        miner: str | None,
        validator: str | None,
    ) -> list[dict]:
        last = int(until // bucket_s)
        first = last - buckets + 1
        since = first * bucket_s
        if validator:
            names = VOTE_POINT
            found = self.validations.vote_series(bucket_s, since, until, validator)
        else:
            names = TASK_POINT
            found = self.validations.task_series(bucket_s, since, until, miner)
        return [
            {"at": bucket * bucket_s, **(found.get(bucket) or dict.fromkeys(names, 0))}
            for bucket in range(first, last + 1)
        ]


def validator_row(
    hotkey: str, totals: dict, standing: dict, last_seen: dict[str, float]
) -> dict:
    row = {
        **dict.fromkeys(VALIDATOR_TOTALS, 0),
        "last_vote_at": None,
        **totals.get(hotkey, {}),
    }
    judged = row["agreed"] + row["disagreed"]
    return {
        "hotkey": hotkey,
        "active": hotkey in last_seen,
        "last_seen": last_seen.get(hotkey),
        **row,
        "agreement": round(row["agreed"] / judged, 4) if judged else None,
        "audits": 0,
        "disagreements": 0,
        "excluded": False,
        **standing.get(hotkey, {}),
    }


def uid_of(registry, hotkey: str | None) -> int | None:
    entry = registry.registered(hotkey)
    return entry.uid if entry else None


def with_uids(registry, row: dict) -> dict:
    """The row with a uid beside each hotkey it names; null for one off the metagraph."""
    return row | {
        f"{name}_uid": uid_of(registry, row[name])
        for name in ("miner", "validator")
        if name in row
    }


def with_keys(registry, row: dict, hotkey: str) -> dict:
    entry = registry.registered(hotkey)
    return row | {
        "uid": entry.uid if entry else None,
        "coldkey": entry.coldkey if entry else None,
    }


async def in_progress(core) -> dict:
    claimed = await core.redis.zrange(queues.CLAIMS, 0, LIVE_ROWS - 1, withscores=True)
    claims = []
    for task_id, expires_at in claimed:
        payload = await core.payload(task_id) or {}
        claims.append(
            {
                "task_id": task_id,
                "kind": payload.get("kind", CRAWL),
                "miner": await core.claim_holder(task_id),
                "urls": payload.get("url_count", len(payload.get("urls", []))),
                "expires_at": expires_at,
            }
        )
    uploads = []
    active = await core.validation.active()
    waiting = await core.validation.open_ids(LIVE_ROWS)
    waiting += await core.validation.seeding_ids(LIVE_ROWS - len(waiting))
    for task_id in waiting:
        job = await core.validation.job(task_id)
        if job is None:
            continue
        voters = await core.validation.voters(task_id)
        uploads.append(
            {
                "task_id": task_id,
                "kind": job.get("kind", CRAWL),
                "miner": job["miner"],
                "urls": len(job["urls"]),
                "completed_at": job["completed_at"],
                "deadline": job.get("deadline"),
                "voters": [
                    {"hotkey": voter, "uid": uid_of(core.registry, voter)}
                    for voter in sorted(voters)
                ],
                "electorate": len(active | voters),
            }
        )
    return {
        "claims": [with_uids(core.registry, claim) for claim in claims],
        "uploads": [with_uids(core.registry, upload) for upload in uploads],
    }


def router(core) -> APIRouter:
    logs: Logs = core.logs
    api = APIRouter()

    def visible() -> float:
        return time.time() - core.ledger_delay

    async def miner_rows(hours: int) -> list[dict]:
        until = visible()
        rows = await logs.read(logs.miner_rows, until - hours * HOUR, until)
        for row in rows:
            row["uid"] = uid_of(core.registry, row["hotkey"])
            row["in_flight"] = await core.tasks[CRAWL].in_flight(row["hotkey"])
            row["waiting"] = await core.tasks[CRAWL].waiting(row["hotkey"])
        return rows

    async def validator_rows(hours: int) -> list[dict]:
        until = visible()
        last_seen = await core.validation.last_seen()
        rows = await logs.read(
            logs.validator_rows, until - hours * HOUR, until, last_seen
        )
        for row in rows:
            row["uid"] = uid_of(core.registry, row["hotkey"])
        return rows

    @api.get("/v1/overview")
    async def overview(hours: Hours = SHARE_WINDOW_H):
        async def make() -> dict:
            miners = await logs.cached(("miners", hours), lambda: miner_rows(hours))
            validators = await logs.cached(
                ("validators", hours), lambda: validator_rows(hours)
            )
            worked = [miner for miner in miners if miner["tasks"]]
            return {
                "as_of": visible(),
                "window_hours": hours,
                "queue": {
                    kind: await tasks.depth() for kind, tasks in core.tasks.items()
                },
                "claimed": int(await core.redis.zcard(queues.CLAIMS)),
                "validating": await core.validation.depth()
                + await core.validation.seeding(),
                "oldest_validation_s": await core.validation.oldest_age(),
                "publishing": await core.publish.depth(),
                "miners": len(worked),
                "validators": {
                    "active": sum(validator["active"] for validator in validators),
                    "known": len(validators),
                },
                "window": {
                    **{
                        name: sum(miner[name] for miner in worked)
                        for name in MINER_TOTALS
                    },
                    "votes": sum(validator["votes"] for validator in validators),
                    "disagreements": sum(
                        validator["disagreed"] for validator in validators
                    ),
                },
                "total": {
                    "void": 0,
                    **await logs.read(logs.validations.verdicts),
                },
            }

        return await logs.cached(("overview", hours), make)

    @api.get("/v1/stats/series")
    async def series(
        bucket_minutes: int = BUCKET_MINUTES[-1],
        buckets: Annotated[int, Query(ge=1, le=MAX_BUCKETS)] = 24,
        miner: Id = None,
        validator: Id = None,
    ):
        if bucket_minutes not in BUCKET_MINUTES:
            raise HTTPException(422, f"bucket_minutes is one of {BUCKET_MINUTES}")
        if miner and validator:
            raise HTTPException(422, "pass a miner or a validator, not both")
        bucket_s = bucket_minutes * 60

        async def make() -> dict:
            points = await logs.read(
                logs.series, bucket_s, buckets, visible(), miner, validator
            )
            return {"bucket_s": bucket_s, "points": points}

        return await logs.cached(("series", bucket_s, buckets, miner, validator), make)

    @api.get("/v1/live")
    async def live():
        return await logs.cached(("live",), lambda: in_progress(core))

    @api.get("/v1/miners")
    async def miners(hours: Hours = SHARE_WINDOW_H):
        rows = await logs.cached(("miners", hours), lambda: miner_rows(hours))
        return {"window_hours": hours, "miners": rows}

    @api.get("/v1/miners/{hotkey}")
    async def miner(hotkey: PathId):
        until = visible()
        pools = {}
        for kind, tasks in core.tasks.items():
            budget = await logs.read(logs.budgets.get, hotkey, kind)
            pools[kind] = {
                "budget": budget.budget,
                "verified": budget.verified,
                "in_flight": await tasks.in_flight(hotkey),
                "waiting": await tasks.waiting(hotkey),
                "locked_until": await logs.read(
                    logs.budgets.locked_until, hotkey, kind
                ),
            }
        totals = await logs.read(
            logs.validations.miner_totals, until - SHARE_WINDOW_H * HOUR, until
        )
        window = {**dict.fromkeys(MINER_TOTALS, 0), **totals.get(hotkey, {})}
        window.pop("last_scored_at", None)
        shares = await logs.read(logs.budgets.shares, SHARE_WINDOW_H, until)
        verdicts = await logs.read(logs.validations.verdicts, hotkey)
        return {
            **with_keys(core.registry, {"hotkey": hotkey}, hotkey),
            "known": bool(
                sum(verdicts.values()) or await logs.read(logs.budgets.pools_of, hotkey)
            ),
            "window_hours": SHARE_WINDOW_H,
            "pools": pools,
            "coverage": (
                await logs.read(logs.budgets.coverage_report, SHARE_WINDOW_H, until)
            ).get(hotkey, {}),
            "verdicts": verdicts,
            "share": shares.get(CRAWL, {}).get(hotkey, 0.0),
            "window": window,
            "transitions": await logs.read(logs.budgets.history, hotkey),
        }

    @api.get("/v1/validators")
    async def validators(hours: Hours = SHARE_WINDOW_H):
        rows = await logs.cached(("validators", hours), lambda: validator_rows(hours))
        return {"window_hours": hours, "validators": rows}

    @api.get("/v1/validators/{hotkey}")
    async def validator(hotkey: PathId, hours: Hours = SHARE_WINDOW_H):
        rows = await logs.cached(("validators", hours), lambda: validator_rows(hours))
        for row in rows:
            if row["hotkey"] == hotkey:
                return {**with_keys(core.registry, row, hotkey), "known": True}
        unknown = validator_row(hotkey, {}, {}, {})
        return {**with_keys(core.registry, unknown, hotkey), "known": False}

    @api.get("/v1/votes")
    async def votes(
        validator: Id = None,
        miner: Id = None,
        task_id: Id = None,
        verdict: Verdict | None = None,
        agreed: bool | None = None,
        before: int | None = None,
        limit: Limit = PAGE,
    ):
        found, following = await logs.read(
            logs.validations.votes,
            validator,
            miner,
            task_id,
            verdict,
            agreed,
            visible(),
            before,
            limit,
        )
        return {
            "votes": [with_uids(core.registry, vote) for vote in found],
            "next": following,
        }

    @api.get("/v1/events")
    async def events(
        miner: Id = None,
        task_id: Id = None,
        outcome: Outcome | None = None,
        before: int | None = None,
        limit: Limit = PAGE,
    ):
        found, following = await logs.read(
            logs.events.events, miner, task_id, outcome, before, limit
        )
        return {"events": found, "next": following}

    @api.get("/v1/tasks")
    async def tasks(
        miner: Id = None,
        validator: Id = None,
        verdict: Verdict | None = None,
        kind: Kind | None = None,
        since: float = 0.0,
        before: float | None = None,
        limit: Limit = PAGE,
    ):
        """Finalized tasks, newest first; pass `next` back as `before` for the next page."""
        before = visible() if before is None else min(before, visible())
        found = await logs.read(
            logs.validations.recent,
            miner,
            validator,
            since,
            before,
            limit,
            verdict,
            kind,
        )
        full = len(found) == limit
        return {
            "tasks": [with_uids(core.registry, task) for task in found],
            "next": found[-1]["scored_at"] if full else None,
        }

    @api.get("/v1/tasks/{task_id}")
    async def task(task_id: PathId):
        uploads = [
            with_uids(core.registry, upload)
            for upload in await logs.read(logs.validations.uploads, task_id)
            if upload["scored_at"] <= visible()
        ]
        scored = uploads[0] if uploads else None
        state = with_uids(core.registry, await task_state(core, task_id, scored))
        if scored is None:
            return {**state, "uploads": [], "votes": [], "urls": []}
        votes = await logs.read(
            logs.validations.votes_on, task_id, scored["upload_key"]
        )
        return {
            **state,
            "uploads": uploads,
            "votes": [with_uids(core.registry, vote) for vote in votes],
            "urls": await logs.read(logs.validations.urls, task_id),
        }

    return api


async def task_state(core, task_id: str, scored: dict | None) -> dict:
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
