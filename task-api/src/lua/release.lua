local held, ready = KEYS[1], KEYS[2]
local due = redis.call('ZRANGEBYSCORE', held, '-inf', ARGV[1], 'LIMIT', 0, tonumber(ARGV[2]))
for _, task_id in ipairs(due) do
  redis.call('ZREM', held, task_id)
  local payload = redis.call('GET', 'task:' .. task_id)
  if payload then
    redis.call('ZADD', ready, cjson.decode(payload)['rank'], task_id)
  end
end
return #due
