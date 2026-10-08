local task_id, validator, now = ARGV[1], ARGV[2], tonumber(ARGV[3])
if redis.call('EXISTS', 'vjob:' .. task_id) == 0 then return 0 end
if not redis.call('ZSCORE', KEYS[1], task_id) then return 0 end
if redis.call('SADD', 'vseen:' .. task_id, validator) == 0 then return 0 end
redis.call('RPUSH', 'votes:' .. task_id, ARGV[4])
redis.call('ZADD', KEYS[2], now, validator)
return redis.call('LLEN', 'votes:' .. task_id)
