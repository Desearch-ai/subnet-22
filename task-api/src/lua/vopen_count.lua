local open = 0
for _, task_id in ipairs(redis.call('SMEMBERS', KEYS[1])) do
  if redis.call('ZSCORE', KEYS[2], task_id) then
    open = open + 1
  else
    redis.call('SREM', KEYS[1], task_id)
  end
end
return open
