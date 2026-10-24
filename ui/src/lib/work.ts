import type { ClaimCounts, HourPoint } from '@/api/types'

export const QUEUE_EMPTY = 'QUEUE_EMPTY'
const MINUTES_PER_HOUR = 60
const SECONDS_PER_MINUTE = 60

export function refused(counts: ClaimCounts, code: string): number {
  return counts.refused[code] ?? 0
}

/** Refusals are logged once a minute per miner and reason, so this is how many miners were turned away each minute. */
export function perMinute(count: number, minutes = MINUTES_PER_HOUR): number {
  return minutes > 0 ? count / minutes : 0
}

/** The hours that are over, and the one still running. */
export function splitHours(points: readonly HourPoint[]): {
  done: HourPoint[]
  current: HourPoint | undefined
} {
  return { done: points.slice(0, -1), current: points.at(-1) }
}

/** Minutes of the point's hour that had passed at `asOf`, at least one. */
export function minutesInto(point: HourPoint, asOf: number): number {
  return Math.min(MINUTES_PER_HOUR, Math.max(1, (asOf - point.at) / SECONDS_PER_MINUTE))
}

/** How many minutes the waiting tasks would last at the pace tasks were done. */
export function queueMinutes(waiting: number, perHour: number): number | null {
  return perHour > 0 ? waiting / (perHour / MINUTES_PER_HOUR) : null
}

export function averagePerHour<P>(points: readonly P[], value: (point: P) => number): number {
  return points.length > 0
    ? points.reduce((sum, point) => sum + value(point), 0) / points.length
    : 0
}

const SAME_WITHIN = 0.05

/** The value against the day's hourly average, in words. */
export function compareToAverage(value: number, average: number): string {
  if (average <= 0) return 'none earlier in the day'
  const change = value / average - 1
  if (Math.abs(change) < SAME_WITHIN) return 'about the same as the 24-hour average'
  return `${Math.round(Math.abs(change) * 100)}% ${change > 0 ? 'above' : 'below'} the 24-hour average`
}
