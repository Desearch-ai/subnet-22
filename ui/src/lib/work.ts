import type { ClaimCounts, HourPoint } from '@/api/types'

export const QUEUE_EMPTY = 'QUEUE_EMPTY'
const MINUTES_PER_HOUR = 60

export function refused(counts: ClaimCounts, code: string): number {
  return counts.refused[code] ?? 0
}

/** The hours that are over, and the one still running. */
export function splitHours(points: readonly HourPoint[]): {
  done: HourPoint[]
  current: HourPoint | undefined
} {
  return { done: points.slice(0, -1), current: points.at(-1) }
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
