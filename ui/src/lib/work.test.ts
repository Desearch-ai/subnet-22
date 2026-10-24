import { describe, expect, it } from 'vitest'
import type { ClaimCounts, HourPoint } from '@/api/types'
import { averagePerHour, queueMinutes, splitHours } from './work'

const claims: ClaimCounts = {
  issued: 1200,
  completed: 1180,
  lapsed: 2,
  requeued: 1,
  dropped: 0,
  refused: { QUEUE_EMPTY: 6000, NO_CAPACITY: 120, LOCKED_OUT: 60, WAITING_FOR_VERDICTS: 30 },
}

function hour(at: number): HourPoint {
  return { ...claims, at, tasks: 0, pass: 0, fail: 0, void: 0, returned: 0, credited: 0 }
}

describe('hours', () => {
  it('keeps the running hour apart from the ones that are over', () => {
    const { done, current } = splitHours([hour(0), hour(3600), hour(7200)])
    expect(done.map((point) => point.at)).toEqual([0, 3600])
    expect(current?.at).toBe(7200)
    expect(splitHours([]).current).toBeUndefined()
  })

  it('says how long the waiting tasks last at the pace tasks were done', () => {
    expect(queueMinutes(300, 18_000)).toBe(1)
    expect(queueMinutes(10, 0)).toBeNull()
  })
})

describe('averagePerHour', () => {
  it('averages the hours, zero when there are none', () => {
    expect(averagePerHour([10, 20, 30], (n) => n)).toBe(20)
    expect(averagePerHour([], (n: number) => n)).toBe(0)
  })
})
