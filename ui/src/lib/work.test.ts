import { describe, expect, it } from 'vitest'
import type { ClaimCounts, HourPoint } from '@/api/types'
import {
  averagePerHour,
  compareToAverage,
  minutesInto,
  perMinute,
  queueMinutes,
  refused,
  splitHours,
} from './work'

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

describe('refusals', () => {
  it('reads one reason, zero for a reason never logged', () => {
    expect(refused(claims, 'QUEUE_EMPTY')).toBe(6000)
    expect(refused(claims, 'ALREADY_HELD')).toBe(0)
  })

  it('turns an hour of refusals into miners turned away each minute', () => {
    expect(perMinute(6000)).toBe(100)
    expect(perMinute(300, 15)).toBe(20)
    expect(perMinute(5, 0)).toBe(0)
  })
})

describe('hours', () => {
  it('keeps the running hour apart from the ones that are over', () => {
    const { done, current } = splitHours([hour(0), hour(3600), hour(7200)])
    expect(done.map((point) => point.at)).toEqual([0, 3600])
    expect(current?.at).toBe(7200)
    expect(splitHours([]).current).toBeUndefined()
  })

  it('counts the minutes of the running hour, at least one and at most sixty', () => {
    expect(minutesInto(hour(3600), 3600 + 900)).toBe(15)
    expect(minutesInto(hour(3600), 3600)).toBe(1)
    expect(minutesInto(hour(0), 99_999)).toBe(60)
  })

  it('says how long the waiting tasks last at the pace tasks were done', () => {
    expect(queueMinutes(300, 18_000)).toBe(1)
    expect(queueMinutes(10, 0)).toBeNull()
  })
})

describe('the last hour against the day', () => {
  it('averages the hours and says how far an hour is from it', () => {
    expect(averagePerHour([10, 20, 30], (n) => n)).toBe(20)
    expect(averagePerHour([], (n: number) => n)).toBe(0)
    expect(compareToAverage(20_400, 20_000)).toBe('about the same as the 24-hour average')
    expect(compareToAverage(15_000, 20_000)).toBe('25% below the 24-hour average')
    expect(compareToAverage(26_000, 20_000)).toBe('30% above the 24-hour average')
    expect(compareToAverage(5, 0)).toBe('none earlier in the day')
  })
})
