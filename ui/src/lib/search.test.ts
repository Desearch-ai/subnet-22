import { describe, expect, it } from 'vitest'
import type { MinerSummary } from '@/api/types'
import { findMiners, looksLikeHotkey } from './search'

function miner(hotkey: string, uid: number | null, share: number): MinerSummary {
  return {
    hotkey,
    uid,
    coldkey: null,
    share,
    budget: 1,
    in_flight: 0,
    waiting: 0,
    locked_until: null,
    verified: 0,
    tasks: 0,
    returned: 0,
    credited: 0,
    coverage: null,
    last_scored_at: null,
    pass: 0,
    fail: 0,
    void: 0,
  }
}

const MINERS = [
  miner('5Fabc', 36, 0.01),
  miner('5Gxyz', 3, 0.02),
  miner('5Habc', 136, 0.05),
  miner('5Jqrs', null, 0.5),
]

describe('findMiners', () => {
  it('matches UIDs by their first digits, the exact one first, then the largest share', () => {
    expect(findMiners(MINERS, '36', 5).map((m) => m.uid)).toEqual([36])
    expect(findMiners(MINERS, '3', 5).map((m) => m.uid)).toEqual([3, 36])
    expect(findMiners(MINERS, '13', 5).map((m) => m.uid)).toEqual([136])
  })

  it('matches any part of a hotkey and stops at the limit', () => {
    expect(findMiners(MINERS, 'abc', 5).map((m) => m.hotkey)).toEqual(['5Habc', '5Fabc'])
    expect(findMiners(MINERS, 'abc', 1)).toHaveLength(1)
    expect(findMiners(MINERS, '  ', 5)).toEqual([])
  })
})

describe('looksLikeHotkey', () => {
  it('takes a 48-character SS58 address', () => {
    expect(looksLikeHotkey('5'.padEnd(48, 'a'))).toBe(true)
    expect(looksLikeHotkey('5abc')).toBe(false)
  })
})
