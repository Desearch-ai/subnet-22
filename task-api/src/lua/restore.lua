redis.call('SET', 'task:' .. ARGV[1], ARGV[2])
redis.call('ZADD', KEYS[1], ARGV[3], ARGV[1])
if cjson.decode(ARGV[2])['kind'] == 'embed' and redis.call('EXISTS', 'embed:pace') == 1 then
  redis.call('HINCRBYFLOAT', 'embed:pace', 'tokens', -1)
end
return redis.call('INCR', 'log:seq')
