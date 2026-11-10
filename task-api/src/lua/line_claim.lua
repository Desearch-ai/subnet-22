local ready, line, alive, hotkey, now = KEYS[1], KEYS[5], KEYS[6], ARGV[1], tonumber(ARGV[3])
local count, keep, slots = math.min(tonumber(ARGV[9]), tonumber(ARGV[12])), tonumber(ARGV[10]), tonumber(ARGV[11])

for _, place in ipairs(redis.call('ZRANGEBYSCORE', alive, '-inf', ARGV[3])) do
  redis.call('ZREM', line, place)
  redis.call('ZREM', alive, place)
end

local places, free = {}, {}
for slot = 0, slots - 1 do
  local place = hotkey .. ':' .. slot
  local since = redis.call('ZSCORE', line, place)
  if since then
    table.insert(places, {place, tonumber(since)})
  else
    table.insert(free, place)
  end
end
table.sort(places, function(a, b) return a[2] < b[2] end)

local function leave(from)
  for at = from, #places do
    redis.call('ZREM', line, places[at][1])
    redis.call('ZREM', alive, places[at][1])
  end
end

local has = redis.call('SCARD', KEYS[3])
local budget = tonumber(ARGV[4])
if has >= budget or redis.call('SCARD', KEYS[4]) >= tonumber(ARGV[8]) then
  leave(1)
  return take(count)
end

local wanted = math.min(count, budget - has)
leave(wanted + 1)
for at = #places + 1, wanted do
  local place = table.remove(free, 1)
  redis.call('ZADD', line, ARGV[3], place)
  places[at] = {place, now}
end

local function waiting()
  return {'line', redis.call('ZRANK', line, places[1][1]), redis.call('ZCARD', line)}
end

local tasks, turn = redis.call('ZCARD', ready), 0
for at = 1, wanted do
  redis.call('ZADD', alive, now + keep, places[at][1])
  if redis.call('ZRANK', line, places[at][1]) < tasks then turn = turn + 1 end
end

-- A line that has left tasks standing longer than a place lasts is not moving, however its front is made up, so whoever asks takes one.
local since = tonumber(redis.call('GET', KEYS[7]))
if tasks == 0 then
  redis.call('DEL', KEYS[7])
  since = nil
elseif not since then
  redis.call('SET', KEYS[7], ARGV[3])
  since = now
end
local stuck = since and now - since > keep
if stuck then turn = math.max(turn, 1) end
if turn == 0 then return waiting() end

local found = take(turn)
if found[1] == 'tasks' then
  -- The line has moved, so a free pass does not outlast the wait that earned it.
  redis.call('DEL', KEYS[7])
  local given = {'tasks'}
  for at = 1, (#found - 1) / 3 do
    for field = 0, 2 do table.insert(given, found[(at - 1) * 3 + 2 + field]) end
    table.insert(given, tostring(places[at][2]))
    redis.call('ZREM', line, places[at][1])
    redis.call('ZREM', alive, places[at][1])
  end
  return given
end
if found[1] == 'held' then
  for at = 1, turn do
    redis.call('ZREM', line, places[at][1])
    redis.call('ZREM', alive, places[at][1])
  end
  local left = redis.call('ZCARD', line)
  return {'line', left, left}
end
return waiting()
