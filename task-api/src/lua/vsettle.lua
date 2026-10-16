local task_id, publish, completed = ARGV[1], ARGV[2], ARGV[3]
local job = redis.call('GET', 'vjob:' .. task_id)
if not job then return nil end
if redis.call('ZREM', KEYS[1], task_id) == 0 then return nil end
local votes = cjson.encode(redis.call('LRANGE', 'votes:' .. task_id, 0, -1))
redis.call('DEL', 'vjob:' .. task_id, 'vseen:' .. task_id, 'votes:' .. task_id)
local decoded = cjson.decode(job)
if decoded['miner'] then
  redis.call('SREM', waiting(decoded['kind'], decoded['miner']), task_id)
end
if publish ~= '' then
  redis.call('SET', 'pjob:' .. task_id, publish)
  redis.call('RPUSH', KEYS[2], task_id)
  redis.call('ZADD', KEYS[3], completed, task_id)
end
return {job, votes}
