"""Files numbered in the order they were written, so a reader follows them one number at a time."""

from __future__ import annotations

import logging

log = logging.getLogger("task_api")


class Feed:
    def __init__(self, name: str):
        self.name = name
        self.counter = f"{name}:seq"
        self.holes = f"{name}:holes"
        self.latest_key = f"{name}/latest.json"

    def seq_key(self, seq: int) -> str:
        return f"{self.name}/seq/{seq:012d}.json"

    async def number(self, storage, redis, key: str, rows: int) -> int:
        """Numbers a file already written; a number is only handed out for a file that exists."""
        at = int(await redis.incr(self.counter))
        try:
            await storage.put_json(self.seq_key(at), {"key": key, "rows": rows})
        except Exception:
            log.exception("could not index %s as %d", key, at)
            await redis.hset(self.holes, at, key)
            return at
        try:
            await storage.put_json(
                self.latest_key, {"seq": at}, cache_control="no-store"
            )
        except Exception:
            log.warning("could not note %d as the newest %s file", at, self.name)
        return at

    async def fill_holes(self, storage, redis) -> int:
        """A number whose index write failed still gets one, so readers never wait on it."""
        filled = 0
        for at, key in (await redis.hgetall(self.holes)).items():
            try:
                await storage.put_json(
                    self.seq_key(int(at)), {"key": key, "rows": None}
                )
            except Exception:
                continue
            await redis.hdel(self.holes, at)
            filled += 1
        return filled
