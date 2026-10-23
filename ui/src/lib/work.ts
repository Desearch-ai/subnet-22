import type { ClaimCounts, HourPoint } from '@/api/types'

export const QUEUE_EMPTY = 'QUEUE_EMPTY'
export const NO_CAPACITY = 'NO_CAPACITY'
const SHOWN_APART: readonly string[] = [QUEUE_EMPTY, NO_CAPACITY]
const MINUTES_PER_HOUR = 60
const SECONDS_PER_MINUTE = 60

export function refused(counts: ClaimCounts, code: string): number {
  return counts.refused[code] ?? 0
}

/** Refusals for every reason but an empty queue and a full budget. */
export function refusedOther(counts: ClaimCounts): number {
  return Object.entries(counts.refused).reduce(
    (sum, [code, n]) => (SHOWN_APART.includes(code) ? sum : sum + (n ?? 0)),
    0,
  )
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

/** How many minutes the waiting tasks would last at the pace they were handed out. */
export function queueMinutes(waiting: number, issuedPerHour: number): number | null {
  return issuedPerHour > 0 ? waiting / (issuedPerHour / MINUTES_PER_HOUR) : null
}
