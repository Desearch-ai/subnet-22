local task_id, holder = ARGV[1], ARGV[2]
if redis.call('GET', 'claim:' .. task_id) ~= holder then return 0 end
local payload = redis.call('GET', 'task:' .. task_id)
local task = payload and cjson.decode(payload) or {}
redis.call('ZREM', KEYS[1], task_id)
redis.call('DEL', 'claim:' .. task_id, 'issued:' .. task_id, 'task:' .. task_id, 'handbacks:' .. task_id)
redis.call('SREM', inflight(task['kind'], holder), task_id)
return 1
