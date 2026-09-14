from __future__ import annotations

import asyncio
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from . import proofs, queues, rounds
from .budget import Budgets, hour_of
from .registry import admins_from_env, receipt_key_from_env, registry_from_env
from .roundlog import RoundLog, receipt_body
from .roundstore import RoundStore
from .seeds import seeds_from_env
from .storage import Storage
from .validations import Validations

MAX_UPLOAD_BYTES = 64_000_000
MAX_BACKLOG_S = 43_200
PAGES_BUCKET = "desearch-pages"
READS_PER_MINUTE = 120
DB_FILE = "task_api.db"


def connect(path: str) -> sqlite3.Connection:
    db = sqlite3.connect(path, check_same_thread=False)
    db.execute("PRAGMA journal_mode=WAL")
    # Receipts are promises to miners, so every commit is synced in full.
    db.execute("PRAGMA synchronous=FULL")
    return db


class Nonces:
    def __init__(self, redis):
        self.redis = redis

    async def claim(self, hotkey: str, nonce: str, ttl: int) -> bool:
        return bool(
            await self.redis.set(f"nonce:{hotkey}:{nonce}", "1", nx=True, ex=ttl)
        )


class State:
    def __init__(self, redis):
        data = Path(os.environ.get("TASK_API_DATA", "."))
        data.mkdir(parents=True, exist_ok=True)

        self.redis = redis
        self.lease_ttl = int(os.environ.get("TASK_API_LEASE_TTL", rounds.LEASE_TTL_S))
        self.validation_ttl = int(os.environ.get("TASK_API_VALIDATION_TTL", "900"))
        self.poll_rate = float(os.environ.get("TASK_API_POLL_RATE", "2"))
        self.reads_per_minute = int(
            os.environ.get("TASK_API_READS_PER_MINUTE", READS_PER_MINUTE)
        )
        self.max_upload = int(
            os.environ.get("TASK_API_MAX_UPLOAD_BYTES", MAX_UPLOAD_BYTES)
        )
        self.max_backlog = float(
            os.environ.get("TASK_API_MAX_BACKLOG_S", MAX_BACKLOG_S)
        )
        self.max_attempts = int(os.environ.get("TASK_API_MAX_ATTEMPTS", "3"))
        self.audit_rate = float(os.environ.get("TASK_API_AUDIT_RATE", "0.05"))
        self.audit_wait = float(os.environ.get("TASK_API_AUDIT_WAIT_S", "3600"))
        self.releases_per_hour = int(os.environ.get("TASK_API_RELEASES_PER_HOUR", "60"))
        self.queue = queues.TaskQueue(redis, self.lease_ttl)
        self.validation = queues.ValidationQueue(
            redis,
            self.validation_ttl,
            int(os.environ.get("TASK_API_VALIDATION_TRIES", "3")),
            int(os.environ.get("TASK_API_VALIDATOR_LEASES", "8")),
            int(os.environ.get("TASK_API_MAX_RELEASES", "3")),
        )
        self.publish = queues.PublishQueue(
            redis,
            int(os.environ.get("TASK_API_PUBLISH_TTL", "600")),
            int(os.environ.get("TASK_API_PUBLISH_TRIES", "5")),
        )
        # Every SQLite call runs on this one thread: off the event loop, and in order.
        self.db_thread = ThreadPoolExecutor(1, thread_name_prefix="sqlite")
        db = connect(str(data / DB_FILE))
        self.budgets = Budgets(db)
        self.log = RoundLog(db)
        self.rounds = RoundStore(db)
        self.validations = Validations(db)
        self.storage = Storage()
        self.pages = Storage(
            bucket=os.environ.get("CF_R2_PAGES_BUCKET", PAGES_BUCKET),
            prefix=os.environ.get("CF_R2_PAGES_PREFIX", ""),
        )
        self.registry = registry_from_env()
        self.admins = admins_from_env()
        self.seeds = seeds_from_env()
        self.current: str | None = self.rounds.latest_revealed()
        self.key = receipt_key_from_env()

    async def db(self, fn, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.db_thread, partial(fn, *args, **kwargs))

    async def record(
        self,
        round_id: str,
        hotkey: str,
        requested_at: float,
        outcome: str,
        seq: int,
        task_id: str | None = None,
        refusal: dict | None = None,
        cause: str | None = None,
    ) -> dict:
        body = receipt_body(
            round_id=round_id,
            hotkey=hotkey,
            requested_at=float(requested_at),
            outcome=outcome,
            seq=seq,
            task_id=task_id,
            refusal=refusal,
            cause=cause,
        )
        signature = self.key.sign(proofs.canonical_json(body)).hex()
        await self.db(
            self.log.record,
            round_id,
            hotkey,
            float(requested_at),
            outcome,
            signature,
            task_id=task_id,
            refusal=refusal,
            seq=seq,
            cause=cause,
        )
        return {"body": body, "signature": signature}

    async def retry_after(self, hotkey: str) -> float | None:
        now = time.time()
        key = f"rl:hotkey:{hotkey}:{int(now)}"
        count = await self.redis.incr(key)
        if count == 1:
            await self.redis.expire(key, 2)
        if count <= self.poll_rate:
            return None
        return max(0.01, round(1 - now % 1, 3))

    async def read_wait(self, ip: str) -> int | None:
        now = time.time()
        key = f"rl:read:{ip}:{int(now // 60)}"
        count = await self.redis.incr(key)
        if count == 1:
            await self.redis.expire(key, 120)
        if count <= self.reads_per_minute:
            return None
        return 60 - int(now % 60)

    async def releases(self, hotkey: str, add: bool = False) -> int:
        key = f"rl:vrelease:{hotkey}:{hour_of()}"
        if not add:
            return int(await self.redis.get(key) or 0)
        count = int(await self.redis.incr(key))
        if count == 1:
            await self.redis.expire(key, 7200)
        return count
