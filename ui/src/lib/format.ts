const SECONDS_PER_MINUTE = 60
const SECONDS_PER_HOUR = 3600
const SECONDS_PER_DAY = 86400
const JUST_NOW_S = 10
const SIMILARITY_DIGITS = 6
const MINUS = '−'

export const EMPTY_VALUE = '–'

const NICE_STEPS = [1, 2, 2.5, 5, 10]

const integerFormat = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 })
const compactFormat = new Intl.NumberFormat('en-US', {
  notation: 'compact',
  maximumFractionDigits: 1,
})
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

export function formatCompact(value: number): string {
  return compactFormat.format(value)
}

/** Ticks from zero at a round step, the top one at or above `highest`. */
export function niceTicks(highest: number, steps = 4): number[] {
  if (highest <= 0) return [0]
  const rough = highest / steps
  const power = 10 ** Math.floor(Math.log10(rough))
  const step = (NICE_STEPS.find((nice) => nice * power >= rough) ?? 10) * power
  return Array.from({ length: Math.ceil(highest / step - 1e-9) + 1 }, (_, i) => i * step)
}

export function ratio(part: number, whole: number): number | null {
  return whole > 0 ? part / whole : null
}

export function formatPercent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined) return EMPTY_VALUE
  const smallest = 10 ** -digits
  if (value > 0 && value * 100 < smallest) return `<${smallest.toFixed(digits)}%`
  return `${(value * 100).toFixed(digits)}%`
}

export function formatScore(value: number | null | undefined): string {
  return value === null || value === undefined ? EMPTY_VALUE : value.toFixed(2)
}

export function formatSimilarity(value: number | null | undefined): string {
  return value === null || value === undefined ? EMPTY_VALUE : value.toFixed(SIMILARITY_DIGITS)
}

export function formatSimilarityMargin(value: number | null | undefined, bar: number): string {
  if (value === null || value === undefined) return EMPTY_VALUE
  const margin = value - bar
  const digits = Math.abs(margin).toFixed(SIMILARITY_DIGITS)
  return Number(digits) === 0 || margin > 0 ? `+${digits}` : `${MINUS}${digits}`
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

export function workSeconds(task: {
  claimed_at: number | null
  completed_at: number | null
}): number | null {
  if (task.claimed_at === null || task.completed_at === null) return null
  return task.completed_at - task.claimed_at
}

export function formatWorkTime(task: {
  claimed_at: number | null
  completed_at: number | null
}): string {
  const seconds = workSeconds(task)
  return seconds === null ? EMPTY_VALUE : formatElapsed(seconds)
}

export function formatRelative(atSeconds: number, nowMs: number): string {
  const delta = nowMs / 1000 - atSeconds
  if (Math.abs(delta) < JUST_NOW_S) return 'just now'
  return delta > 0 ? `${formatDuration(delta)} ago` : `in ${formatDuration(-delta)}`
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
