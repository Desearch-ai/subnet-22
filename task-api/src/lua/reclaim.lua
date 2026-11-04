local task_id, now = ARGV[1], tonumber(ARGV[2])
local expiry = redis.call('ZSCORE', KEYS[2], task_id)
-- A completion that arrives in the deadline's own block still counts, so the claim ends only after it.
if not expiry or tonumber(expiry) >= now then return nil end
-- A completion that arrived in time is still being processed.
if redis.call('EXISTS', 'completing:' .. task_id) == 1 then return nil end
local holder = redis.call('GET', 'claim:' .. task_id) or ''
