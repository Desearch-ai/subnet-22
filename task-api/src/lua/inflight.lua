local function inflight(kind, hotkey)
  if not kind or kind == 'crawl' then return 'inflight:' .. hotkey end
  return 'inflight:' .. kind .. ':' .. hotkey
end
local function waiting(kind, hotkey)
  if not kind or kind == 'crawl' then return 'waiting:' .. hotkey end
  return 'waiting:' .. kind .. ':' .. hotkey
end
