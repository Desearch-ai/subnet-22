"""Redis-only queue. A host is fetchable by one miner at a time."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

QUEUE = "queue:ready"
LEASES = "leases:expiry"
SCAN_DEPTH = 32

CLAIM = """
local queue, leases = KEYS[1], KEYS[2]
local hotkey, ttl, depth, now = ARGV[1], tonumber(ARGV[2]), tonumber(ARGV[3]), tonumber(ARGV[4])

local candidates = redis.call('ZRANGE', queue, 0, depth - 1)
for i = 1, #candidates do
  local task_id = candidates[i]
  local payload = redis.call('GET', 'task:' .. task_id)
  if payload then
    local hosts = cjson.decode(payload)['hosts']
    local taken = {}
    local ok = true
    for j = 1, #hosts do
      local key = 'host:lock:' .. hosts[j]
      if redis.call('SET', key, task_id, 'NX', 'EX', ttl) then
        taken[#taken + 1] = key
      else
        ok = false
        break
      end
    end
    if ok then
      redis.call('ZREM', queue, task_id)
      redis.call('ZADD', leases, now + ttl, task_id)
      redis.call('SET', 'lease:' .. task_id, hotkey)
      redis.call('SADD', 'inflight:' .. hotkey, task_id)
      return {task_id, payload, redis.call('INCR', 'log:seq')}
    end
    for j = 1, #taken do redis.call('DEL', taken[j]) end
  else
    redis.call('ZREM', queue, task_id)
  end
end
return nil
"""

RELEASE = """
local task_id, hotkey = ARGV[1], ARGV[2]
local payload = redis.call('GET', 'task:' .. task_id)
if payload then
  local hosts = cjson.decode(payload)['hosts']
  for i = 1, #hosts do
    local key = 'host:lock:' .. hosts[i]
    if redis.call('GET', key) == task_id then redis.call('DEL', key) end
  end
end
redis.call('ZREM', KEYS[1], task_id)
redis.call('DEL', 'lease:' .. task_id)
if hotkey ~= '' then redis.call('SREM', 'inflight:' .. hotkey, task_id) end
return redis.call('INCR', 'log:seq')
"""


@dataclass
class Lease:
    task_id: str
    payload: dict
    expires_at: float
    seq: int


class Refusal(Exception):
    def __init__(self, code: str, **inputs):
        self.code = code
        self.inputs = inputs
        super().__init__(code)

    def as_dict(self) -> dict:
        return {"code": self.code, "inputs": self.inputs}


class Queue:
    def __init__(self, redis, lease_ttl: int):
        self.redis = redis
        self.lease_ttl = lease_ttl
        self._claim = None
        self._release = None

    async def register(self) -> None:
        self._claim = self.redis.register_script(CLAIM)
        self._release = self.redis.register_script(RELEASE)

    async def fill(
        self, round_id: str, order: list[str], payloads: dict[str, dict]
    ) -> int:
        pipe = self.redis.pipeline()
        for position, task_id in enumerate(order):
            payload = dict(payloads[task_id], round_id=round_id, position=position)
            pipe.set(f"task:{task_id}", json.dumps(payload))
            pipe.zadd(QUEUE, {task_id: position})
        await pipe.execute()
        return len(order)

    async def depth(self) -> int:
        return int(await self.redis.zcard(QUEUE))

    async def in_flight(self, hotkey: str) -> int:
        return int(await self.redis.scard(f"inflight:{hotkey}"))

    async def lease(self, hotkey: str, budget: int) -> Lease:
        held = await self.in_flight(hotkey)
        if held >= budget:
            raise Refusal("NO_CAPACITY", budget=budget, in_flight=held)
        if await self.depth() == 0:
            raise Refusal("QUEUE_EMPTY", depth=0)

        now = time.time()
        result = await self._claim(
            keys=[QUEUE, LEASES],
            args=[hotkey, self.lease_ttl, SCAN_DEPTH, now],
        )
        if not result:
            raise Refusal("HOST_LOCKED", scanned=SCAN_DEPTH, depth=await self.depth())

        task_id, payload, seq = result
        return Lease(
            task_id=task_id if isinstance(task_id, str) else task_id.decode(),
            payload=json.loads(payload),
            expires_at=now + self.lease_ttl,
            seq=int(seq),
        )

    async def holder(self, task_id: str) -> str | None:
        value = await self.redis.get(f"lease:{task_id}")
        return value if isinstance(value, str) or value is None else value.decode()

    async def release(self, task_id: str, hotkey: str = "") -> int:
        return int(await self._release(keys=[LEASES], args=[task_id, hotkey]))

    async def next_seq(self) -> int:
        return int(await self.redis.incr("log:seq"))

    async def expired(self, now: float | None = None) -> list[str]:
        now = now or time.time()
        found = await self.redis.zrangebyscore(LEASES, "-inf", now)
        return [item if isinstance(item, str) else item.decode() for item in found]

    async def requeue(self, task_id: str) -> int:
        payload = await self.redis.get(f"task:{task_id}")
        if not payload:
            return 0
        position = json.loads(payload).get("position", 0)
        holder = await self.holder(task_id)
        seq = await self.release(task_id, holder or "")
        await self.redis.zadd(QUEUE, {task_id: position})
        return seq
