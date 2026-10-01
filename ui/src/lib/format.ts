const SECONDS_PER_MINUTE = 60
const SECONDS_PER_HOUR = 3600
const SECONDS_PER_DAY = 86400
const JUST_NOW_S = 10

export const EMPTY_VALUE = '–'

const integerFormat = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 })
const absoluteFormat = new Intl.DateTimeFormat('en-GB', {
  year: 'numeric',
  month: 'short',
  day: '2-digit',
  hour: '2-digit',
  minute: '2-digit',
  second: '2-digit',
  timeZoneName: 'short',
})
const clockFormat = new Intl.DateTimeFormat('en-GB', { hour: '2-digit', minute: '2-digit' })

export function formatInt(value: number | null | undefined): string {
  return value === null || value === undefined ? EMPTY_VALUE : integerFormat.format(value)
}

export function ratio(part: number, whole: number): number | null {
  return whole > 0 ? part / whole : null
}

export function formatPercent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined) return EMPTY_VALUE
  return `${(value * 100).toFixed(digits)}%`
}

export function formatScore(value: number | null | undefined): string {
  return value === null || value === undefined ? EMPTY_VALUE : value.toFixed(2)
}

export function formatDuration(seconds: number): string {
  const total = Math.max(0, Math.round(seconds))
  if (total < SECONDS_PER_MINUTE) return `${total} s`
  if (total < SECONDS_PER_HOUR) return `${Math.floor(total / SECONDS_PER_MINUTE)} min`
  if (total < SECONDS_PER_DAY) {
    const hours = Math.floor(total / SECONDS_PER_HOUR)
    const minutes = Math.floor((total % SECONDS_PER_HOUR) / SECONDS_PER_MINUTE)
    return minutes === 0 ? `${hours} h` : `${hours} h ${minutes} min`
  }
  const days = Math.floor(total / SECONDS_PER_DAY)
  const hours = Math.floor((total % SECONDS_PER_DAY) / SECONDS_PER_HOUR)
  return hours === 0 ? `${days} d` : `${days} d ${hours} h`
}

export function formatElapsed(seconds: number): string {
  const total = Math.max(0, Math.round(seconds))
  if (total < SECONDS_PER_MINUTE) return `${total} s`
  if (total < SECONDS_PER_HOUR) {
    return `${Math.floor(total / SECONDS_PER_MINUTE)} min ${total % SECONDS_PER_MINUTE} s`
  }
  return formatDuration(total)
}

export function crawlSeconds(task: {
  claimed_at: number | null
  completed_at: number | null
}): number | null {
  if (task.claimed_at === null || task.completed_at === null) return null
  return task.completed_at - task.claimed_at
}

export function formatCrawlTime(task: {
  claimed_at: number | null
  completed_at: number | null
}): string {
  const seconds = crawlSeconds(task)
  return seconds === null ? EMPTY_VALUE : formatElapsed(seconds)
}

export function formatRelative(atSeconds: number, nowMs: number): string {
  const delta = nowMs / 1000 - atSeconds
  if (Math.abs(delta) < JUST_NOW_S) return 'just now'
  return delta > 0 ? `${formatDuration(delta)} ago` : `in ${formatDuration(-delta)}`
}

export function formatCountdown(untilSeconds: number, nowMs: number): string {
  const left = Math.max(0, Math.round(untilSeconds - nowMs / 1000))
  const minutes = Math.floor(left / SECONDS_PER_MINUTE)
  const seconds = left % SECONDS_PER_MINUTE
  return `${minutes}:${String(seconds).padStart(2, '0')}`
}

export function formatAbsolute(atSeconds: number): string {
  return absoluteFormat.format(new Date(atSeconds * 1000))
}

export function formatClock(atSeconds: number): string {
  return clockFormat.format(new Date(atSeconds * 1000))
}

export function truncateMiddle(value: string, head = 6, tail = 6): string {
  return value.length <= head + tail + 1 ? value : `${value.slice(0, head)}…${value.slice(-tail)}`
}
