import { describe, expect, it } from 'vitest'
import {
  EMPTY_VALUE,
  formatCountdown,
  formatDuration,
  formatInt,
  formatPercent,
  formatRelative,
  ratio,
  truncateMiddle,
} from './format'

const NOW_MS = 1_790_000_000_000
const NOW_S = NOW_MS / 1000

describe('formatInt', () => {
  it('groups thousands and shows a dash for missing values', () => {
    expect(formatInt(12000)).toBe('12,000')
    expect(formatInt(null)).toBe(EMPTY_VALUE)
  })
})

describe('ratio and formatPercent', () => {
  it('returns null when the whole is zero', () => {
    expect(ratio(3, 0)).toBeNull()
    expect(formatPercent(ratio(3, 0))).toBe(EMPTY_VALUE)
  })

  it('formats a share as a percentage', () => {
    expect(formatPercent(ratio(35, 40))).toBe('87.5%')
    expect(formatPercent(0.41, 0)).toBe('41%')
  })
})

describe('formatDuration', () => {
  it('picks the largest sensible unit', () => {
    expect(formatDuration(14.2)).toBe('14 s')
    expect(formatDuration(125)).toBe('2 min')
    expect(formatDuration(3600)).toBe('1 h')
    expect(formatDuration(3900)).toBe('1 h 5 min')
    expect(formatDuration(90000)).toBe('1 d 1 h')
  })
})

describe('formatRelative', () => {
  it('describes past and future times', () => {
    expect(formatRelative(NOW_S - 3, NOW_MS)).toBe('just now')
    expect(formatRelative(NOW_S - 120, NOW_MS)).toBe('2 min ago')
    expect(formatRelative(NOW_S + 7200, NOW_MS)).toBe('in 2 h')
  })
})

describe('formatCountdown', () => {
  it('counts down in minutes and seconds and stops at zero', () => {
    expect(formatCountdown(NOW_S + 412, NOW_MS)).toBe('6:52')
    expect(formatCountdown(NOW_S - 5, NOW_MS)).toBe('0:00')
  })
})

describe('truncateMiddle', () => {
  it('keeps short values and shortens long ones', () => {
    expect(truncateMiddle('ab12')).toBe('ab12')
    expect(truncateMiddle('5FGtU35pohrno7p2m5ex9jv9pbzAVGiGSZijvTFknLfu4N9q')).toBe('5FGtU3…fu4N9q')
  })
})
