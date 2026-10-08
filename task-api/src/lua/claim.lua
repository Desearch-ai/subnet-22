local queue, claims, mine, unscored = KEYS[1], KEYS[2], KEYS[3], KEYS[4]
local hotkey, ttl, now = ARGV[1], tonumber(ARGV[2]), tonumber(ARGV[3])
local budget, page, most = tonumber(ARGV[4]), tonumber(ARGV[5]), tonumber(ARGV[7])

local held = redis.call('SCARD', mine)
if held >= budget then return {'full', tostring(held)} end
local waiting = redis.call('SCARD', unscored)
if waiting >= tonumber(ARGV[8]) then return {'waiting', tostring(waiting)} end
local wanted = math.min(tonumber(ARGV[9]), budget - held)
local taken = {'tasks'}
local seen, start = 0, 0
while start < most and (#taken - 1) / 3 < wanted do
  local batch = redis.call('ZRANGE', queue, start, start + page - 1)
  if #batch == 0 then break end
  local removed = 0
  for _, task_id in ipairs(batch) do
    if (#taken - 1) / 3 >= wanted then break end
    local payload = redis.call('GET', 'task:' .. task_id)
    if not payload then
      redis.call('ZREM', queue, task_id)
      removed = removed + 1
    elseif redis.call('SISMEMBER', 'holders:' .. task_id, hotkey) == 1 then
      seen = seen + 1
    else
      redis.call('ZREM', queue, task_id)
      removed = removed + 1
      redis.call('ZADD', claims, now + ttl, task_id)
      redis.call('SET', 'claim:' .. task_id, hotkey)
      redis.call('SADD', mine, task_id)
      redis.call('SADD', 'holders:' .. task_id, hotkey)
      redis.call('EXPIRE', 'holders:' .. task_id, tonumber(ARGV[6]))
      table.insert(taken, task_id)
      table.insert(taken, payload)
      table.insert(taken, redis.call('INCR', 'log:seq'))
    end
  end
  start = start + #batch - removed
end
if #taken > 1 then return taken end
if seen > 0 then return {'held', tostring(seen)} end
return {'empty'}
