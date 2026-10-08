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
redis.call('ZADD', KEYS[2], tonumber(ARGV[6]), task_id)
local kind = cjson.decode(job)['kind']
redis.call('SMOVE', inflight(kind, hotkey), waiting(kind, hotkey), task_id)
return redis.call('INCR', 'log:seq')
