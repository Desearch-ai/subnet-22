if redis.call('ZREM', KEYS[1], ARGV[1]) == 0 then return 0 end
if redis.call('EXISTS', 'pjob:' .. ARGV[1]) == 0 then return 0 end
if redis.call('INCR', 'ptries:' .. ARGV[1]) >= tonumber(ARGV[2]) then
  redis.call('ZREM', KEYS[3], ARGV[1])
  redis.call('SADD', KEYS[4], ARGV[1])
  return -1
end
redis.call('RPUSH', KEYS[2], ARGV[1])
return 1
