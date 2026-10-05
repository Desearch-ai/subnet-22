"""Every completed crawl upload, written in numbered signed files to the uploads bucket, so validators count each miner's work themselves."""

from __future__ import annotations

import json
import time

from desearch.manifest import UPLOAD_LOG_LATEST, log_payload, upload_log_key

PENDING = "uploadlog:pending"
NEXT = "uploadlog:next"
# How many pending entries the file being written holds, so a retry writes the same ones.
WRITING = "uploadlog:writing"
BATCH = 1000


def entry(job: dict) -> dict:
    reported = job.get("reported") or {}
    return {
        "task_id": job["task_id"],
        "miner": job["miner"],
        "key": job["key"],
        "completed_at": job["completed_at"],
        "assigned": len(set(job["urls"])),
        "rows": int(reported.get("rows", 0)),
        "ok": int(reported.get("ok", 0)),
        "errors": int(reported.get("errors", 0)),
    }


async def note(redis, job: dict) -> None:
    await redis.rpush(PENDING, json.dumps(entry(job)))


async def flush(storage, redis, key) -> int | None:
    """Writes the pending entries as the next numbered file; on a failed write they wait for the next pass."""
    writing = int(await redis.get(WRITING) or 0)
    pending = await redis.lrange(PENDING, 0, (writing or BATCH) - 1)
    if not pending:
        return None
    if not writing:
        await redis.set(WRITING, len(pending))
    seq = int(await redis.get(NEXT) or 1)
    body = {
        "seq": seq,
        "written_at": time.time(),
        "entries": [json.loads(raw) for raw in pending],
        "signer": key.ss58_address,
    }
    body["signature"] = key.sign(log_payload(body)).hex()
    await storage.put_json(upload_log_key(seq), body)
    await storage.put_json(UPLOAD_LOG_LATEST, {"seq": seq}, cache_control="no-store")
    pipe = redis.pipeline()
    pipe.ltrim(PENDING, len(pending), -1)
    pipe.set(NEXT, seq + 1)
    pipe.delete(WRITING)
    await pipe.execute()
    return seq
