import { describe, expect, it } from 'vitest'
import { compareValues, nextSort, sortRows } from './sort'

interface Row {
  name: string
  share: number | null
}

const ROWS: Row[] = [
  { name: 'b', share: 0.2 },
  { name: 'a', share: null },
  { name: 'c', share: 0.7 },
]

const valueOf = (row: Row, key: keyof Row) => row[key]

describe('compareValues', () => {
  it('orders missing values first', () => {
    expect(compareValues(null, 1)).toBeLessThan(0)
    expect(compareValues(2, 1)).toBeGreaterThan(0)
    expect(compareValues('a', 'b')).toBeLessThan(0)
  })
})

describe('sortRows', () => {
  it('sorts in both directions without changing the input', () => {
    const descending = sortRows(ROWS, { key: 'share', direction: 'desc' }, valueOf)
    expect(descending.map((row) => row.name)).toEqual(['c', 'b', 'a'])
    const ascending = sortRows(ROWS, { key: 'name', direction: 'asc' }, valueOf)
    expect(ascending.map((row) => row.name)).toEqual(['a', 'b', 'c'])
    expect(ROWS.map((row) => row.name)).toEqual(['b', 'a', 'c'])
  })
})

describe('nextSort', () => {
  it('flips the direction on the same key and starts descending on a new one', () => {
    expect(nextSort({ key: 'share', direction: 'desc' }, 'share').direction).toBe('asc')
    expect(nextSort({ key: 'share', direction: 'asc' }, 'name')).toEqual({
      key: 'name',
      direction: 'desc',
    })
  })
})
