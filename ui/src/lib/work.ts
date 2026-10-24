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
