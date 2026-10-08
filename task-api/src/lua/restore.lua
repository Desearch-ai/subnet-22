redis.call('SET', 'task:' .. ARGV[1], ARGV[2])
redis.call('ZADD', KEYS[1], ARGV[3], ARGV[1])
return redis.call('INCR', 'log:seq')
