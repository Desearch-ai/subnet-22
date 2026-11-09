local held, ready, pace = KEYS[1], KEYS[2], KEYS[3]
local now, target, step, burst, most = tonumber(ARGV[1]), tonumber(ARGV[2]), tonumber(ARGV[3]), tonumber(ARGV[4]), tonumber(ARGV[5])
local kept = redis.call('HMGET', pace, 'rate', 'at', 'tokens', 'released')
local rate, at, tokens = tonumber(kept[1]) or target, tonumber(kept[2]) or now, tonumber(kept[3]) or 0
local released = tonumber(kept[4]) or rate
local since = math.max(0, now - at)
local change = rate * step * since / 3600
if rate < target then
  rate = math.min(target, rate + change)
elseif rate > target then
  rate = math.max(target, rate - change)
end
local limit = math.max(1, rate * burst / 3600)
tokens = math.min(limit, math.max(-rate / 6, tokens) + since * rate / 3600)
local moved = 0
local whole = math.min(math.floor(tokens), most, math.floor(limit) - redis.call('ZCARD', ready))
if whole > 0 then
  local due = redis.call('ZRANGEBYSCORE', held, '-inf', ARGV[1], 'LIMIT', 0, whole)
  for _, task_id in ipairs(due) do
    redis.call('ZREM', held, task_id)
    local payload = redis.call('GET', 'task:' .. task_id)
    if payload then
      redis.call('ZADD', ready, cjson.decode(payload)['rank'], task_id)
      moved = moved + 1
    end
  end
end
released = released * math.exp(-since / 600) + moved * 6
redis.call('HSET', pace, 'rate', tostring(rate), 'at', now >= at and ARGV[1] or tostring(at), 'tokens', tostring(tokens - moved), 'released', tostring(released))
return {moved, tostring(rate)}
