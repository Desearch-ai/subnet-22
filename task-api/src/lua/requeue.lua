redis.call('ZREM', KEYS[2], task_id)
redis.call('DEL', 'claim:' .. task_id, 'issued:' .. task_id)
local payload = redis.call('GET', 'task:' .. task_id)
local task = payload and cjson.decode(payload) or {}
if holder ~= '' then redis.call('SREM', inflight(task['kind'], holder), task_id) end
if payload then
  redis.call('ZADD', KEYS[1], task['rank'] or task['position'] or 0, task_id)
end
if task['kind'] == 'embed' and ARGV[3] ~= 'untold' then
  redis.call('INCR', 'handbacks:' .. task_id)
  redis.call('EXPIRE', 'handbacks:' .. task_id, 172800)
  if payload and redis.call('EXISTS', 'embed:pace') == 1 then
    redis.call('HINCRBYFLOAT', 'embed:pace', 'tokens', -1)
  end
end
local seq = redis.call('INCR', 'log:seq')
