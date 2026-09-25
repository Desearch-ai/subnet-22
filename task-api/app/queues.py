from __future__ import annotations

import json
import time
from dataclasses import dataclass

QUEUE = "queue:ready"
LEASES = "leases:expiry"
VALIDATE = "validate:ready"
VLEASES = "vleases:expiry"
COMPLETED = "vjobs:completed"
AUDITS = "vjobs:audits"
PUBLISH = "publish:ready"
PLEASES = "publish:leases"
PPENDING = "publish:pending"
PDEAD = "publish:dead"
ROUNDS = "rounds:seq"
# Published pages waiting to become embed batches, pushed by the publisher.
EMBED_INPUTS = "embed:inputs"
ROUND_SPAN = 10_000_000
CLAIM_SCAN = 50
HOLDERS_TTL_S = 7 * 86400
KINDS = ("crawl", "embed")


def ready_key(kind: str) -> str:
    return QUEUE if kind == "crawl" else f"{QUEUE}:{kind}"


def validate_key(kind: str) -> str:
    return VALIDATE if kind == "crawl" else f"{VALIDATE}:{kind}"


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
local queue, leases, mine = KEYS[1], KEYS[2], KEYS[3]
local hotkey, ttl, now = ARGV[1], tonumber(ARGV[2]), tonumber(ARGV[3])

local held = redis.call('SCARD', mine)
if held >= tonumber(ARGV[4]) then return {'full', tostring(held)} end
local seen = 0
for _, task_id in ipairs(redis.call('ZRANGE', queue, 0, tonumber(ARGV[5]) - 1)) do
  local payload = redis.call('GET', 'task:' .. task_id)
  if not payload then
    redis.call('ZREM', queue, task_id)
  elseif redis.call('SISMEMBER', 'holders:' .. task_id, hotkey) == 1 then
    seen = seen + 1
  else
    redis.call('ZREM', queue, task_id)
    redis.call('ZADD', leases, now + ttl, task_id)
    redis.call('SET', 'lease:' .. task_id, hotkey)
    redis.call('SADD', mine, task_id)
    redis.call('SADD', 'holders:' .. task_id, hotkey)
    redis.call('EXPIRE', 'holders:' .. task_id, tonumber(ARGV[6]))
    return {'task', task_id, payload, redis.call('INCR', 'log:seq')}
  end
end
if seen > 0 then return {'held', tostring(seen)} end
return {'empty'}
"""

REQUEUE = """
redis.call('ZREM', KEYS[2], task_id)
redis.call('DEL', 'lease:' .. task_id, 'issued:' .. task_id)
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
local holder = redis.call('GET', 'lease:' .. task_id) or ''
"""
    + REQUEUE
    + "return {holder, seq}"
)

ABANDON = (
    INFLIGHT
    + """
local task_id, holder = ARGV[1], ARGV[2]
if redis.call('GET', 'lease:' .. task_id) ~= holder then return nil end
"""
    + REQUEUE
    + "return seq"
)

# Kept inflight until its verdict, so budgets bound unscored work.
COMPLETE = """
local task_id, hotkey, job, now = ARGV[1], ARGV[2], ARGV[3], tonumber(ARGV[4])
if redis.call('GET', 'lease:' .. task_id) ~= hotkey then return nil end
local issued = redis.call('GET', 'issued:' .. task_id)
if not issued or cjson.decode(issued)['key'] ~= ARGV[5] then return nil end
local expiry = redis.call('ZSCORE', KEYS[1], task_id)
if not expiry or tonumber(expiry) < now then return nil end
if redis.call('EXISTS', 'vjob:' .. task_id) == 1 then return nil end
redis.call('ZREM', KEYS[1], task_id)
redis.call('DEL', 'lease:' .. task_id, 'issued:' .. task_id, 'task:' .. task_id)
redis.call('SET', 'vjob:' .. task_id, job)
redis.call('RPUSH', KEYS[2], task_id)
redis.call('ZADD', KEYS[3], now, task_id)
return redis.call('INCR', 'log:seq')
"""

RESTORE = """
redis.call('SET', 'task:' .. ARGV[1], ARGV[2])
redis.call('ZADD', KEYS[1], ARGV[3], ARGV[1])
return redis.call('INCR', 'log:seq')
"""

# Skip jobs this validator voted on, so audits go to someone else.
VCLAIM = """
local validator, expires, cap = ARGV[1], ARGV[2], tonumber(ARGV[3])
local held = 'vheld:' .. validator
if redis.call('SCARD', held) >= cap then return {'full'} end
local skipped, found = {}, nil
for _ = 1, tonumber(ARGV[4]) do
  local task_id = redis.call('LPOP', KEYS[1])
  if not task_id then break end
  local job = redis.call('GET', 'vjob:' .. task_id)
  if job then
    if redis.call('SISMEMBER', 'vseen:' .. task_id, validator) == 1 then
      table.insert(skipped, task_id)
    else
      redis.call('SET', 'vlease:' .. task_id, validator)
      redis.call('ZADD', KEYS[2], expires, task_id)
      redis.call('SADD', held, task_id)
      found = job
      break
    end
  end
end
for i = #skipped, 1, -1 do redis.call('LPUSH', KEYS[1], skipped[i]) end
if found then return {'job', found} end
return {'empty'}
"""

VBEGIN = """
local task_id, validator, now = ARGV[1], ARGV[2], tonumber(ARGV[3])
if redis.call('GET', 'vlease:' .. task_id) ~= validator then return nil end
local expiry = tonumber(redis.call('ZSCORE', KEYS[1], task_id) or 0)
if expiry < now then return nil end
redis.call('ZADD', KEYS[1], math.max(expiry, tonumber(ARGV[4])), task_id)
return redis.call('GET', 'vjob:' .. task_id)
"""

VVOTE = """
local task_id, validator = ARGV[1], ARGV[2]
if redis.call('GET', 'vlease:' .. task_id) ~= validator then return 0 end
redis.call('DEL', 'vlease:' .. task_id)
redis.call('ZREM', KEYS[1], task_id)
redis.call('SREM', 'vheld:' .. validator, task_id)
redis.call('RPUSH', 'votes:' .. task_id, ARGV[3])
redis.call('SADD', 'vseen:' .. task_id, validator)
redis.call('RPUSH', KEYS[2], task_id)
redis.call('ZADD', KEYS[3], ARGV[4], task_id)
return 1
"""

FINAL = """
local job = redis.call('GET', 'vjob:' .. task_id)
if not job then return nil end
local votes = cjson.encode(redis.call('LRANGE', 'votes:' .. task_id, 0, -1))
redis.call('DEL', 'vlease:' .. task_id, 'vjob:' .. task_id, 'vtries:' .. task_id,
  'vreleases:' .. task_id, 'vseen:' .. task_id, 'votes:' .. task_id)
redis.call('ZREM', KEYS[1], task_id)
redis.call('ZREM', KEYS[2], task_id)
redis.call('ZREM', KEYS[3], task_id)
redis.call('LREM', KEYS[4], 0, task_id)
local decoded = cjson.decode(job)
if decoded['miner'] then
  redis.call('SREM', inflight(decoded['kind'], decoded['miner']), task_id)
end
"""

VFINAL = (
    INFLIGHT
    + """
local task_id, validator, publish, completed = ARGV[1], ARGV[2], ARGV[3], ARGV[4]
local holder = redis.call('GET', 'vlease:' .. task_id)
if validator == '' then
  if holder or redis.call('ZSCORE', KEYS[1], task_id) then return nil end
elseif holder ~= validator then
  return nil
end
"""
    + FINAL
    + """
if validator ~= '' then redis.call('SREM', 'vheld:' .. validator, task_id) end
if publish ~= '' then
  redis.call('SET', 'pjob:' .. task_id, publish)
  redis.call('RPUSH', KEYS[5], task_id)
  redis.call('ZADD', KEYS[6], completed, task_id)
end
return {job, votes}
"""
)

VRELEASE = (
    INFLIGHT
    + """
local task_id, validator, counted, limit = ARGV[1], ARGV[2], ARGV[3] == '1', tonumber(ARGV[4])
if redis.call('GET', 'vlease:' .. task_id) ~= validator then return nil end
redis.call('SREM', 'vheld:' .. validator, task_id)
if not counted or redis.call('INCR', 'vreleases:' .. task_id) < limit then
  redis.call('DEL', 'vlease:' .. task_id)
  redis.call('ZREM', KEYS[1], task_id)
  redis.call('RPUSH', KEYS[4], task_id)
  return {'requeued'}
end
"""
    + FINAL
    + "return {'exhausted', job, votes}"
)

VEXPIRE = """
local task_id = ARGV[1]
if redis.call('ZREM', KEYS[1], task_id) == 0 then return 0 end
local holder = redis.call('GET', 'vlease:' .. task_id)
if holder then redis.call('SREM', 'vheld:' .. holder, task_id) end
redis.call('DEL', 'vlease:' .. task_id)
local tries = redis.call('INCR', 'vtries:' .. task_id)
if tries < tonumber(ARGV[2]) then redis.call('RPUSH', KEYS[2], task_id) end
return tries
"""

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

# Set aside a job that keeps failing so it stops blocking leases.
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
class Lease:
    task_id: str
    payload: dict
    expires_at: float
    seq: int


@dataclass
class Settled:
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
    def __init__(self, redis, lease_ttl: int, kind: str = "crawl"):
        self.redis = redis
        self.lease_ttl = lease_ttl
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

    async def lease(self, hotkey: str, budget: int) -> Lease:
        now = time.time()
        status, *found = await self._claim(
            keys=[self.ready, LEASES, inflight_key(self.kind, hotkey)],
            args=[hotkey, self.lease_ttl, now, budget, CLAIM_SCAN, HOLDERS_TTL_S],
        )
        status = _text(status)
        if status == "full":
            raise Refusal("NO_CAPACITY", budget=budget, in_flight=int(found[0]))
        if status == "held":
            raise Refusal("ALREADY_HELD", depth=await self.depth(), held=int(found[0]))
        if status == "empty":
            raise Refusal("QUEUE_EMPTY", depth=await self.depth())
        task_id, payload, seq = found
        return Lease(
            task_id=_text(task_id),
            payload=json.loads(payload),
            expires_at=now + self.lease_ttl,
            seq=int(seq),
        )

    async def lease_holder(self, task_id: str) -> str | None:
        return _text(await self.redis.get(f"lease:{task_id}"))

    async def complete(
        self, task_id: str, hotkey: str, job: dict, upload_key: str
    ) -> int | None:
        seq = await self._complete(
            keys=[LEASES, validate_key(self.kind), COMPLETED],
            args=[task_id, hotkey, json.dumps(job), time.time(), upload_key],
        )
        return None if seq is None else int(seq)

    async def abandon(self, task_id: str, hotkey: str) -> int | None:
        seq = await self._abandon(keys=[self.ready, LEASES], args=[task_id, hotkey])
        return None if seq is None else int(seq)

    async def reclaim(
        self, task_id: str, now: float | None = None
    ) -> tuple[str, int] | None:
        found = await self._reclaim(
            keys=[self.ready, LEASES], args=[task_id, now or time.time()]
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
            _text(item) for item in await self.redis.zrangebyscore(LEASES, "-inf", now)
        ]


class ValidationQueue:
    def __init__(
        self,
        redis,
        lease_ttl: int,
        max_tries: int = 3,
        max_leases: int = 8,
        max_releases: int = 3,
    ):
        self.redis = redis
        self.lease_ttl = lease_ttl
        self.max_tries = max_tries
        self.max_leases = max_leases
        self.max_releases = max_releases
        self._claim = redis.register_script(VCLAIM)
        self._begin = redis.register_script(VBEGIN)
        self._vote = redis.register_script(VVOTE)
        self._finalize = redis.register_script(VFINAL)
        self._release = redis.register_script(VRELEASE)
        self._expire = redis.register_script(VEXPIRE)

    async def depth(self) -> int:
        return sum([int(await self.redis.llen(validate_key(k))) for k in KINDS])

    async def active(self) -> int:
        return int(await self.redis.zcard(VLEASES))

    async def _waiting_list(self, task_id: str) -> str:
        """The list a job waits in, which depends on the kind of task it judges."""
        return validate_key((await self.job(task_id) or {}).get("kind", "crawl"))

    async def job(self, task_id: str) -> dict | None:
        found = await self.redis.get(f"vjob:{task_id}")
        return json.loads(found) if found else None

    async def votes(self, task_id: str) -> list[dict]:
        return [
            json.loads(v) for v in await self.redis.lrange(f"votes:{task_id}", 0, -1)
        ]

    async def oldest_age(self) -> float:
        oldest = await self.redis.zrange(COMPLETED, 0, 0, withscores=True)
        return round(time.time() - oldest[0][1], 1) if oldest else 0.0

    async def lease_holder(self, task_id: str) -> str | None:
        return _text(await self.redis.get(f"vlease:{task_id}"))

    async def lease(
        self, validator: str, kinds: tuple[str, ...] = ("crawl",)
    ) -> tuple[dict, float] | None:
        expires_at = time.time() + self.lease_ttl
        for kind in kinds:
            status, *found = await self._claim(
                keys=[validate_key(kind), VLEASES],
                args=[validator, expires_at, self.max_leases, CLAIM_SCAN],
            )
            if _text(status) == "full":
                raise Refusal("LEASE_LIMIT", held=self.max_leases)
            if found:
                return json.loads(found[0]), expires_at
        return None

    async def begin(self, task_id: str, validator: str, grace: float) -> dict | None:
        now = time.time()
        job = await self._begin(
            keys=[VLEASES], args=[task_id, validator, now, now + grace]
        )
        return json.loads(job) if job else None

    async def vote(
        self, task_id: str, validator: str, vote: dict, deadline: float
    ) -> bool:
        return bool(
            await self._vote(
                keys=[VLEASES, await self._waiting_list(task_id), AUDITS],
                args=[task_id, validator, json.dumps(vote), deadline],
            )
        )

    async def finalize(
        self, task_id: str, validator: str, publish: dict | None = None
    ) -> Settled | None:
        """An empty validator settles a job nobody holds."""
        found = await self._finalize(
            keys=[
                VLEASES,
                COMPLETED,
                AUDITS,
                await self._waiting_list(task_id),
                PUBLISH,
                PPENDING,
            ],
            args=[
                task_id,
                validator,
                json.dumps(publish) if publish else "",
                (publish or {}).get("completed_at") or time.time(),
            ],
        )
        if not found:
            return None
        job, votes = found
        return Settled(json.loads(job), _votes(votes))

    async def settle(self, task_id: str) -> Settled | None:
        return await self.finalize(task_id, "")

    async def release(
        self, task_id: str, validator: str, counted: bool = True
    ) -> tuple[str, Settled | None] | None:
        found = await self._release(
            keys=[VLEASES, COMPLETED, AUDITS, await self._waiting_list(task_id)],
            args=[task_id, validator, "1" if counted else "0", self.max_releases],
        )
        if not found:
            return None
        if _text(found[0]) == "requeued":
            return "requeued", None
        return "exhausted", Settled(json.loads(found[1]), _votes(found[2]))

    async def expired(self, now: float | None = None) -> list[str]:
        now = now or time.time()
        return [
            _text(item) for item in await self.redis.zrangebyscore(VLEASES, "-inf", now)
        ]

    async def overdue_audits(self, now: float | None = None) -> list[str]:
        now = now or time.time()
        return [
            _text(item) for item in await self.redis.zrangebyscore(AUDITS, "-inf", now)
        ]

    async def give_back(self, task_id: str) -> int:
        return int(
            await self._expire(
                keys=[VLEASES, await self._waiting_list(task_id)],
                args=[task_id, self.max_tries],
            )
        )


class PublishQueue:
    def __init__(self, redis, lease_ttl: int = 600, max_tries: int = 5):
        self.redis = redis
        self.lease_ttl = lease_ttl
        self.max_tries = max_tries
        self._claim = redis.register_script(PCLAIM)
        self._ack = redis.register_script(PACK)
        self._return = redis.register_script(PRETURN)

    async def claim(self, count: int) -> list[dict]:
        jobs = await self._claim(
            keys=[PUBLISH, PLEASES], args=[count, time.time() + self.lease_ttl]
        )
        return [json.loads(job) for job in jobs or []]

    async def ack(self, task_id: str) -> None:
        await self._ack(keys=[PLEASES, PPENDING], args=[task_id])

    async def extend_lease(self, task_id: str) -> None:
        await self.redis.zadd(PLEASES, {task_id: time.time() + self.lease_ttl}, xx=True)

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
            _text(item) for item in await self.redis.zrangebyscore(PLEASES, "-inf", now)
        ]

    async def give_back(self, task_id: str) -> int:
        return int(
            await self._return(
                keys=[PLEASES, PUBLISH, PPENDING, PDEAD], args=[task_id, self.max_tries]
            )
        )

    async def depth(self) -> int:
        return int(await self.redis.llen(PUBLISH))

    async def oldest_age(self) -> float:
        oldest = await self.redis.zrange(PPENDING, 0, 0, withscores=True)
        return round(time.time() - oldest[0][1], 1) if oldest else 0.0
