from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from . import flow, proofs, queues, rounds, sampling
from .budget import Budgets
from .checks import Checks
from .embeddings import Embeddings
from .logs import Logs
from .registry import admins_from_env, receipt_key_from_env, registry_from_env
from .roundlog import RoundLog, receipt_body
from .roundstore import RoundStore
from .seeds import seeds_from_env
from .storage import Storage
from .validations import Validations

MAX_UPLOAD_BYTES = 100_000_000
MAX_BACKLOG_S = 43_200
QUEUE_TARGET = 1_200
PAGES_BUCKET = "desearch-pages"
READS_PER_MINUTE = 120
LOG_READS_PER_MINUTE = 60
POLL_RATE = 0.5
POLL_WINDOW_S = 10
FAILED_WRITES_PER_MINUTE = 30
MINUTE = 60
DB_FILE = "task_api.db"
DEFAULT_EMBED_MODEL = "qwen3-embedding-8b"


class Connection(sqlite3.Connection):
    """Commits inside a batch wait for the batch, so a final_verdict lands whole or not at all."""

    batching = False

    def commit(self) -> None:
        if not self.batching:
            super().commit()

    @contextlib.contextmanager
    def batch(self):
        self.batching = True
        try:
            yield
        except BaseException:
            self.rollback()
            raise
        finally:
            self.batching = False
        super().commit()


def connect(path: str) -> Connection:
    db = sqlite3.connect(path, check_same_thread=False, factory=Connection)
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
        self.claim_ttl = int(os.environ.get("TASK_API_CLAIM_TTL", rounds.CLAIM_TTL_S))
        self.validation_ttl = int(os.environ.get("TASK_API_VALIDATION_TTL", "900"))
        self.poll_rate = float(os.environ.get("TASK_API_POLL_RATE", POLL_RATE))
        self.reads_per_minute = int(
            os.environ.get("TASK_API_READS_PER_MINUTE", READS_PER_MINUTE)
        )
        self.log_reads_per_minute = int(
            os.environ.get("TASK_API_LOG_READS_PER_MINUTE", LOG_READS_PER_MINUTE)
        )
        self.failed_writes_per_minute = int(
            os.environ.get(
                "TASK_API_FAILED_WRITES_PER_MINUTE", FAILED_WRITES_PER_MINUTE
            )
        )
        self.max_upload = int(
            os.environ.get("TASK_API_MAX_UPLOAD_BYTES", MAX_UPLOAD_BYTES)
        )
        self.max_backlog = float(
            os.environ.get("TASK_API_MAX_BACKLOG_S", MAX_BACKLOG_S)
        )
        self.max_attempts = int(os.environ.get("TASK_API_MAX_ATTEMPTS", "3"))
        self.ledger_delay = float(os.environ.get("TASK_API_LEDGER_DELAY_S", "0"))
        self.check_share = float(os.environ.get("TASK_API_CHECK_SHARE", sampling.SHARE))
        self.queue_target = int(os.environ.get("TASK_API_QUEUE_TARGET", QUEUE_TARGET))
        self.open_listed: tuple[str, ...] | None = None
        self.publish_rate = flow.PublishRate(
            float(os.environ.get("TASK_API_PUBLISH_LAG_S", flow.LAG_TARGET_S)),
            float(os.environ.get("TASK_API_PUBLISH_LAG_LIMIT_S", flow.LAG_LIMIT_S)),
        )
        # Each open upload's manifest and deadline, read once instead of on every listing.
        self.open_manifests: dict[str, tuple[dict, float]] = {}
        self.tasks = {
            kind: queues.TaskQueue(redis, self.claim_ttl + rounds.UPLOAD_GRACE_S, kind)
            for kind in queues.KINDS
        }
        # Off until Desearch's own model ships; on, it runs the stand-in for testing.
        self.embed_tasks = os.environ.get("TASK_API_EMBED_TASKS", "0") == "1"
        self.embed_model = os.environ.get("TASK_API_EMBED_MODEL", DEFAULT_EMBED_MODEL)
        self.validation = queues.ValidationQueue(
            redis, int(os.environ.get("TASK_API_ACTIVE_S", queues.ACTIVE_S))
        )
        self.publish = queues.PublishQueue(
            redis,
            int(os.environ.get("TASK_API_PUBLISH_TTL", "600")),
            int(os.environ.get("TASK_API_PUBLISH_TRIES", "5")),
        )
        # Every SQLite call runs on this one thread: off the event loop, and in order.
        self.db_thread = ThreadPoolExecutor(1, thread_name_prefix="sqlite")
        db = self.sqlite = connect(str(data / DB_FILE))
        self.budgets = Budgets(db)
        self.log = RoundLog(db)
        self.rounds = RoundStore(db)
        self.validations = Validations(db)
        self.checks = Checks(db)
        self.embeddings = Embeddings(db)
        self.logs = Logs(connect(str(data / DB_FILE)))
        self.storage = Storage()
        self.pages = Storage(
            bucket=os.environ.get("CF_R2_PAGES_BUCKET", PAGES_BUCKET),
            prefix=os.environ.get("CF_R2_PAGES_PREFIX", ""),
        )
        self.registry = registry_from_env()
        self.admins = admins_from_env()
        self.seeds = seeds_from_env()
        self.current: dict[str, str] = self.rounds.latest_revealed()
        self.key = receipt_key_from_env()

    async def payload(self, task_id: str) -> dict | None:
        found = await self.redis.get(f"task:{task_id}")
        return json.loads(found) if found else None

    async def claim_holder(self, task_id: str) -> str | None:
        return await self.redis.get(f"claim:{task_id}")

    async def next_seq(self) -> int:
        return int(await self.redis.incr("log:seq"))

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
        block: int | None = None,
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
            block=block,
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
            block=block,
        )
        return {"body": body, "signature": signature}

    async def retry_after(self, hotkey: str) -> float | None:
        now = time.time()
        window = int(now // POLL_WINDOW_S)
        key = f"rl:hotkey:{hotkey}:{window}"
        count = await self.redis.incr(key)
        if count == 1:
            await self.redis.expire(key, 2 * POLL_WINDOW_S)
        if count <= max(1, round(self.poll_rate * POLL_WINDOW_S)):
            return None
        return max(0.01, round((window + 1) * POLL_WINDOW_S - now, 3))

    async def first_refusal(self, hotkey: str, code: str) -> bool:
        """Only the first of a run of identical refusals is signed into the log."""
        key = f"refused:{hotkey}:{code}:{int(time.time() // MINUTE)}"
        return bool(await self.redis.set(key, "1", nx=True, ex=2 * MINUTE))

    async def read_wait(self, ip: str, logs: bool) -> int | None:
        """Log reads have a budget of their own, so a dashboard cannot use up the rest."""
        now = time.time()
        kind, limit = (
            ("logs", self.log_reads_per_minute)
            if logs
            else ("read", self.reads_per_minute)
        )
        key = f"rl:{kind}:{ip}:{int(now // MINUTE)}"
        count = await self.redis.incr(key)
        if count == 1:
            await self.redis.expire(key, 2 * MINUTE)
        if count <= limit:
            return None
        return MINUTE - int(now % MINUTE)

    async def write_wait(self, ip: str) -> int | None:
        """Seconds an address that kept failing authentication waits before it may write again."""
        now = time.time()
        failed = await self.redis.get(f"rl:denied:{ip}:{int(now // MINUTE)}")
        if int(failed or 0) < self.failed_writes_per_minute:
            return None
        return MINUTE - int(now % MINUTE)

    async def write_denied(self, ip: str) -> None:
        key = f"rl:denied:{ip}:{int(time.time() // MINUTE)}"
        if await self.redis.incr(key) == 1:
            await self.redis.expire(key, 2 * MINUTE)
