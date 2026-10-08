local task_id, holder = ARGV[1], ARGV[2]
if redis.call('GET', 'claim:' .. task_id) ~= holder then return nil end
