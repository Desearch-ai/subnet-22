local hotkey, expiry = ARGV[1], tonumber(ARGV[2])
for i = 3, #ARGV do
  if redis.call('GET', 'claim:' .. ARGV[i]) == hotkey then
    redis.call('ZADD', KEYS[1], 'XX', expiry, ARGV[i])
  end
end
return 1
