local held, ready, ranks = KEYS[1], KEYS[2], KEYS[3]
local due = redis.call('ZRANGEBYSCORE', held, '-inf', ARGV[1], 'LIMIT', 0, tonumber(ARGV[2]))
for _, task_id in ipairs(due) do
  redis.call('ZREM', held, task_id)
  local rank = redis.call('HGET', ranks, task_id)
  if rank then
    redis.call('HDEL', ranks, task_id)
    if redis.call('EXISTS', 'task:' .. task_id) == 1 then redis.call('ZADD', ready, rank, task_id) end
  else
    local payload = redis.call('GET', 'task:' .. task_id)
    if payload then redis.call('ZADD', ready, cjson.decode(payload)['rank'], task_id) end
  end
end
return #due
