local jobs = {}
while #jobs < tonumber(ARGV[1]) do
  local task_id = redis.call('LPOP', KEYS[1])
  if not task_id then break end
  local job = redis.call('GET', 'pjob:' .. task_id)
  if job then
    redis.call('ZADD', KEYS[2], ARGV[2], task_id)
    table.insert(jobs, job)
  end
end
return jobs
