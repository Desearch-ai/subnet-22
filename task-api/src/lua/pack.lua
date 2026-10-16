redis.call('ZREM', KEYS[1], ARGV[1])
if redis.call('ZREM', KEYS[2], ARGV[1]) == 1 then redis.call('INCR', KEYS[3]) end
return redis.call('DEL', 'pjob:' .. ARGV[1], 'ptries:' .. ARGV[1])
