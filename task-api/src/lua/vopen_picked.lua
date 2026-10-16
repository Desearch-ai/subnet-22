if redis.call('ZREM', KEYS[1], ARGV[1]) == 0 then return 0 end
redis.call('SET', 'vjob:' .. ARGV[1], ARGV[2])
redis.call('ZADD', KEYS[2], tonumber(ARGV[3]), ARGV[1])
return 1
