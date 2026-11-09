local hotkey, expiry = ARGV[1], tonumber(ARGV[2])
local held = {}
for i = 3, #ARGV, 2 do
  if redis.call('GET', 'claim:' .. ARGV[i]) == hotkey then
    redis.call('ZADD', KEYS[1], 'XX', expiry, ARGV[i])
    redis.call('SET', 'issued:' .. ARGV[i], ARGV[i + 1])
    table.insert(held, ARGV[i])
  end
end
return held
