from __future__ import annotations

import json
import time
from dataclasses import dataclass

QUEUE = "queue:ready"
CLAIMS = "claims:expiry"
# Uploads open for every validator to score, by completion time.
VOPEN = "vjobs:open"
# Validators by the time they last asked for work or voted.
VACTIVE = "validators:active"
PUBLISH = "publish:ready"
PCLAIMS = "publish:claims"
PPENDING = "publish:pending"
PDEAD = "publish:dead"
ROUNDS = "rounds:seq"
# Published pages waiting to become embed batches, pushed by the publisher.
EMBED_INPUTS = "embed:inputs"
ROUND_SPAN = 10_000_000
CLAIM_SCAN = 50
OPEN_SCAN = 500
OPEN_BATCH = 8
ACTIVE_S = 3600
# A miner that held the whole front of the queue is still served from further back.
CLAIM_SCAN_MAX = 1000
HOLDERS_TTL_S = 7 * 86400
KINDS = ("crawl", "embed")


def ready_key(kind: str) -> str:
    return QUEUE if kind == "crawl" else f"{QUEUE}:{kind}"


def inflight_key(kind: str, hotkey: str) -> str:
    return f"inflight:{hotkey}" if kind == "crawl" else f"inflight:{kind}:{hotkey}"


# Mirrors inflight_key for scripts that only learn the kind from the task they touch.
INFLIGHT = """
local function inflight(kind, hotkey)
  if not kind or kind == 'crawl' then return 'inflight:' .. hotkey end
  return 'inflight:' .. kind .. ':' .. hotkey
end
"""

# A miner never gets a task it held before, so no one miner can fail a task until it drops.
CLAIM = """
local queue, claims, mine = KEYS[1], KEYS[2], KEYS[3]
local hotkey, ttl, now = ARGV[1], tonumber(ARGV[2]), tonumber(ARGV[3])

local held = redis.call('SCARD', mine)
if held >= tonumber(ARGV[4]) then return {'full', tostring(held)} end
local seen, start, page, most = 0, 0, tonumber(ARGV[5]), tonumber(ARGV[7])
while start < most do
  local batch = redis.call('ZRANGE', queue, start, start + page - 1)
  if #batch == 0 then break end
  local removed = 0
  for _, task_id in ipairs(batch) do
    local payload = redis.call('GET', 'task:' .. task_id)
    if not payload then
      redis.call('ZREM', queue, task_id)
      removed = removed + 1
    elseif redis.call('SISMEMBER', 'holders:' .. task_id, hotkey) == 1 then
      seen = seen + 1
    else
      redis.call('ZREM', queue, task_id)
      redis.call('ZADD', claims, now + ttl, task_id)
      redis.call('SET', 'claim:' .. task_id, hotkey)
      redis.call('SADD', mine, task_id)
      redis.call('SADD', 'holders:' .. task_id, hotkey)
      redis.call('EXPIRE', 'holders:' .. task_id, tonumber(ARGV[6]))
      return {'task', task_id, payload, redis.call('INCR', 'log:seq')}
    end
  end
  start = start + #batch - removed
end
if seen > 0 then return {'held', tostring(seen)} end
return {'empty'}
"""

REQUEUE = """
redis.call('ZREM', KEYS[2], task_id)
redis.call('DEL', 'claim:' .. task_id, 'issued:' .. task_id)
local payload = redis.call('GET', 'task:' .. task_id)
local task = payload and cjson.decode(payload) or {}
if holder ~= '' then redis.call('SREM', inflight(task['kind'], holder), task_id) end
if payload then
  redis.call('ZADD', KEYS[1], task['rank'] or task['position'] or 0, task_id)
end
local seq = redis.call('INCR', 'log:seq')
"""

RECLAIM = (
    INFLIGHT
    + """
local task_id, now = ARGV[1], tonumber(ARGV[2])
local expiry = redis.call('ZSCORE', KEYS[2], task_id)
if not expiry or tonumber(expiry) > now then return nil end
local holder = redis.call('GET', 'claim:' .. task_id) or ''
"""
    + REQUEUE
    + "return {holder, seq}"
)

ABANDON = (
    INFLIGHT
    + """
local task_id, holder = ARGV[1], ARGV[2]
if redis.call('GET', 'claim:' .. task_id) ~= holder then return nil end
"""
    + REQUEUE
    + "return seq"
)

# Kept inflight until its verdict, so budgets bound unscored work.
COMPLETE = """
local task_id, hotkey, job, now = ARGV[1], ARGV[2], ARGV[3], tonumber(ARGV[4])
if redis.call('GET', 'claim:' .. task_id) ~= hotkey then return nil end
local issued = redis.call('GET', 'issued:' .. task_id)
if not issued or cjson.decode(issued)['key'] ~= ARGV[5] then return nil end
local expiry = redis.call('ZSCORE', KEYS[1], task_id)
if not expiry or tonumber(expiry) < now then return nil end
if redis.call('EXISTS', 'vjob:' .. task_id) == 1 then return nil end
redis.call('ZREM', KEYS[1], task_id)
redis.call('DEL', 'claim:' .. task_id, 'issued:' .. task_id, 'task:' .. task_id)
redis.call('SET', 'vjob:' .. task_id, job)
redis.call('ZADD', KEYS[2], now, task_id)
return redis.call('INCR', 'log:seq')
"""

RESTORE = """
redis.call('SET', 'task:' .. ARGV[1], ARGV[2])
redis.call('ZADD', KEYS[1], ARGV[3], ARGV[1])
return redis.call('INCR', 'log:seq')
"""

# One vote per validator per open upload; a vote also marks the validator active.
VVOTE = """
local task_id, validator, now = ARGV[1], ARGV[2], tonumber(ARGV[3])
if redis.call('EXISTS', 'vjob:' .. task_id) == 0 then return 0 end
if not redis.call('ZSCORE', KEYS[1], task_id) then return 0 end
if redis.call('SADD', 'vseen:' .. task_id, validator) == 0 then return 0 end
redis.call('RPUSH', 'votes:' .. task_id, ARGV[4])
redis.call('ZADD', KEYS[2], now, validator)
return redis.call('LLEN', 'votes:' .. task_id)
"""

VSETTLE = (
    INFLIGHT
    + """
local task_id, publish, completed = ARGV[1], ARGV[2], ARGV[3]
local job = redis.call('GET', 'vjob:' .. task_id)
if not job then return nil end
if redis.call('ZREM', KEYS[1], task_id) == 0 then return nil end
local votes = cjson.encode(redis.call('LRANGE', 'votes:' .. task_id, 0, -1))
redis.call('DEL', 'vjob:' .. task_id, 'vseen:' .. task_id, 'votes:' .. task_id)
local decoded = cjson.decode(job)
if decoded['miner'] then
  redis.call('SREM', inflight(decoded['kind'], decoded['miner']), task_id)
end
if publish ~= '' then
  redis.call('SET', 'pjob:' .. task_id, publish)
  redis.call('RPUSH', KEYS[2], task_id)
  redis.call('ZADD', KEYS[3], completed, task_id)
end
return {job, votes}
"""
)

PCLAIM = """
local jobs = {}
while #jobs < tonumber(ARGV[1]) do
  local task_id = redis.call('LPOP', KEYS[1])
  if not task_id then break end
  local job = redis.call('GET', 'pjob:' .. task_id)
  if job then
    redis.call('ZADD', KEYS[2], ARGV[2], task_id)
    table.insert(jobs, job)
  end
end
return jobs
"""

PACK = """
redis.call('ZREM', KEYS[1], ARGV[1])
redis.call('ZREM', KEYS[2], ARGV[1])
return redis.call('DEL', 'pjob:' .. ARGV[1], 'ptries:' .. ARGV[1])
"""

# Set aside a job that keeps failing so it stops blocking claims.
PRETURN = """
if redis.call('ZREM', KEYS[1], ARGV[1]) == 0 then return 0 end
if redis.call('EXISTS', 'pjob:' .. ARGV[1]) == 0 then return 0 end
if redis.call('INCR', 'ptries:' .. ARGV[1]) >= tonumber(ARGV[2]) then
  redis.call('ZREM', KEYS[3], ARGV[1])
  redis.call('SADD', KEYS[4], ARGV[1])
  return -1
end
redis.call('RPUSH', KEYS[2], ARGV[1])
return 1
"""


def _text(value) -> str | None:
    return value.decode() if isinstance(value, bytes) else value


def _votes(encoded) -> list[dict]:
    decoded = json.loads(_text(encoded))
    return [json.loads(vote) for vote in decoded] if isinstance(decoded, list) else []


@dataclass
class Claim:
    task_id: str
    payload: dict
    expires_at: float
    seq: int


@dataclass
class Finalized:
    job: dict
    votes: list[dict]


class Refusal(Exception):
    def __init__(self, code: str, **inputs):
        self.code = code
        self.inputs = inputs
        super().__init__(code)

    def as_dict(self) -> dict:
        return {"code": self.code, "inputs": self.inputs}


class TaskQueue:
    def __init__(self, redis, claim_ttl: int, kind: str = "crawl"):
        self.redis = redis
        self.claim_ttl = claim_ttl
        self.kind = kind
        self.ready = ready_key(kind)
        self._claim = redis.register_script(CLAIM)
        self._reclaim = redis.register_script(RECLAIM)
        self._abandon = redis.register_script(ABANDON)
        self._complete = redis.register_script(COMPLETE)
        self._restore = redis.register_script(RESTORE)

    async def fill(
        self, round_id: str, order: list[str], payloads: dict[str, dict]
    ) -> int:
        base = int(await self.redis.incr(ROUNDS)) * ROUND_SPAN
        pipe = self.redis.pipeline()
        for position, task_id in enumerate(order):
            rank = base + position
            payload = dict(
                payloads[task_id],
                kind=self.kind,
                round_id=round_id,
                position=position,
                rank=rank,
            )
            pipe.set(f"task:{task_id}", json.dumps(payload))
            pipe.zadd(self.ready, {task_id: rank})
        await pipe.execute()
        return len(order)

    async def depth(self) -> int:
        return int(await self.redis.zcard(self.ready))

    async def in_flight(self, hotkey: str) -> int:
        return int(await self.redis.scard(inflight_key(self.kind, hotkey)))

    async def payload(self, task_id: str) -> dict | None:
        found = await self.redis.get(f"task:{task_id}")
        return json.loads(found) if found else None

    async def claim(self, hotkey: str, budget: int) -> Claim:
        now = time.time()
        status, *found = await self._claim(
            keys=[self.ready, CLAIMS, inflight_key(self.kind, hotkey)],
            args=[
                hotkey,
                self.claim_ttl,
                now,
                budget,
                CLAIM_SCAN,
                HOLDERS_TTL_S,
                CLAIM_SCAN_MAX,
            ],
        )
        status = _text(status)
        if status == "full":
            raise Refusal("NO_CAPACITY", budget=budget, in_flight=int(found[0]))
        if status == "held":
            raise Refusal("ALREADY_HELD", depth=await self.depth(), held=int(found[0]))
        if status == "empty":
            raise Refusal("QUEUE_EMPTY", depth=await self.depth())
        task_id, payload, seq = found
        return Claim(
            task_id=_text(task_id),
            payload=json.loads(payload),
            expires_at=now + self.claim_ttl,
            seq=int(seq),
        )

    async def claim_holder(self, task_id: str) -> str | None:
        return _text(await self.redis.get(f"claim:{task_id}"))

    async def complete(
        self, task_id: str, hotkey: str, job: dict, upload_key: str
    ) -> int | None:
        seq = await self._complete(
            keys=[CLAIMS, VOPEN],
            args=[task_id, hotkey, json.dumps(job), time.time(), upload_key],
        )
        return None if seq is None else int(seq)

    async def abandon(self, task_id: str, hotkey: str) -> int | None:
        seq = await self._abandon(keys=[self.ready, CLAIMS], args=[task_id, hotkey])
        return None if seq is None else int(seq)

    async def reclaim(
        self, task_id: str, now: float | None = None
    ) -> tuple[str, int] | None:
        found = await self._reclaim(
            keys=[self.ready, CLAIMS], args=[task_id, now or time.time()]
        )
        if not found:
            return None
        holder, seq = found
        return _text(holder), int(seq)

    async def restore(self, task_id: str, payload: dict) -> int:
        rank = payload.get("rank", payload.get("position", 0))
        return int(
            await self._restore(
                keys=[self.ready], args=[task_id, json.dumps(payload), rank]
            )
        )

    async def next_seq(self) -> int:
        return int(await self.redis.incr("log:seq"))

    async def expired(self, now: float | None = None) -> list[str]:
        now = now or time.time()
        return [
            _text(item) for item in await self.redis.zrangebyscore(CLAIMS, "-inf", now)
        ]


class ValidationQueue:
    """Every validator scores every open upload; an upload closes when it is finalized."""

    def __init__(self, redis, active_s: int = ACTIVE_S):
        self.redis = redis
        self.active_s = active_s
        self._vote = redis.register_script(VVOTE)
        self._finalize = redis.register_script(VSETTLE)

    async def depth(self) -> int:
        return int(await self.redis.zcard(VOPEN))

    async def job(self, task_id: str) -> dict | None:
        found = await self.redis.get(f"vjob:{task_id}")
        return json.loads(found) if found else None

    async def votes(self, task_id: str) -> list[dict]:
        return [
            json.loads(v) for v in await self.redis.lrange(f"votes:{task_id}", 0, -1)
        ]

    async def voters(self, task_id: str) -> set[str]:
        return {_text(v) for v in await self.redis.smembers(f"vseen:{task_id}")}

    async def oldest_age(self) -> float:
        oldest = await self.redis.zrange(VOPEN, 0, 0, withscores=True)
        return round(time.time() - oldest[0][1], 1) if oldest else 0.0

    async def open_ids(self, limit: int = OPEN_SCAN) -> list[str]:
        return [_text(t) for t in await self.redis.zrange(VOPEN, 0, limit - 1)]

    async def open(
        self,
        validator: str,
        kinds: tuple[str, ...] = ("crawl",),
        skip: tuple[str, ...] = (),
        limit: int = OPEN_BATCH,
    ) -> list[dict]:
        """The oldest open uploads of these kinds this validator has not voted on."""
        found = []
        for task_id in await self.open_ids():
            if task_id in skip:
                continue
            job = await self.job(task_id)
            if job is None or job.get("kind", "crawl") not in kinds:
                continue
            if await self.redis.sismember(f"vseen:{task_id}", validator):
                continue
            found.append(job)
            if len(found) >= limit:
                break
        return found

    async def vote(
        self, task_id: str, validator: str, vote: dict, now: float | None = None
    ) -> int:
        """How many votes the upload has now, or zero when this one was not taken."""
        return int(
            await self._vote(
                keys=[VOPEN, VACTIVE],
                args=[task_id, validator, now or time.time(), json.dumps(vote)],
            )
        )

    async def present(self, validator: str, now: float | None = None) -> None:
        """A validator asking for work is in the electorate, however slow its verdicts."""
        await self.redis.zadd(VACTIVE, {validator: now or time.time()})

    async def active(self, now: float | None = None) -> set[str]:
        """Validators that asked for work or voted within the activity window."""
        now = now or time.time()
        await self.redis.zremrangebyscore(VACTIVE, "-inf", now - self.active_s)
        return {_text(v) for v in await self.redis.zrange(VACTIVE, 0, -1)}

    async def finalize(
        self, task_id: str, publish: dict | None = None
    ) -> Finalized | None:
        found = await self._finalize(
            keys=[VOPEN, PUBLISH, PPENDING],
            args=[
                task_id,
                json.dumps(publish) if publish else "",
                (publish or {}).get("completed_at") or time.time(),
            ],
        )
        if not found:
            return None
        job, votes = found
        return Finalized(json.loads(job), _votes(votes))


class PublishQueue:
    def __init__(self, redis, claim_ttl: int = 600, max_tries: int = 5):
        self.redis = redis
        self.claim_ttl = claim_ttl
        self.max_tries = max_tries
        self._claim = redis.register_script(PCLAIM)
        self._ack = redis.register_script(PACK)
        self._return = redis.register_script(PRETURN)

    async def claim(self, count: int) -> list[dict]:
        jobs = await self._claim(
            keys=[PUBLISH, PCLAIMS], args=[count, time.time() + self.claim_ttl]
        )
        return [json.loads(job) for job in jobs or []]

    async def ack(self, task_id: str) -> None:
        await self._ack(keys=[PCLAIMS, PPENDING], args=[task_id])

    async def extend_claim(self, task_id: str) -> None:
        await self.redis.zadd(PCLAIMS, {task_id: time.time() + self.claim_ttl}, xx=True)

    async def mark_lost(self, task_id: str) -> None:
        await self.redis.incr("publish:lost")
        await self.redis.sadd("publish:lost:tasks", task_id)

    async def next_seq(self) -> int:
        return int(await self.redis.incr("changes:seq"))

    async def push_embed_input(self, entry: dict) -> None:
        await self.redis.rpush(EMBED_INPUTS, json.dumps(entry))

    async def dead_count(self) -> int:
        return int(await self.redis.scard(PDEAD))

    async def lost_count(self) -> int:
        return int(await self.redis.get("publish:lost") or 0)

    async def expired(self, now: float | None = None) -> list[str]:
        now = now or time.time()
        return [
            _text(item) for item in await self.redis.zrangebyscore(PCLAIMS, "-inf", now)
        ]

    async def give_back(self, task_id: str) -> int:
        return int(
            await self._return(
                keys=[PCLAIMS, PUBLISH, PPENDING, PDEAD], args=[task_id, self.max_tries]
            )
        )

    async def depth(self) -> int:
        return int(await self.redis.llen(PUBLISH))

    async def oldest_age(self) -> float:
        oldest = await self.redis.zrange(PPENDING, 0, 0, withscores=True)
        return round(time.time() - oldest[0][1], 1) if oldest else 0.0
