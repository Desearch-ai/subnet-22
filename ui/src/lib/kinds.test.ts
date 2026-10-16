import { describe, expect, it } from 'vitest'
import { countColumns, formatTaskSize, readKind } from './kinds'
import { withKind } from './paths'

describe('readKind', () => {
  it('accepts the two kinds and nothing else', () => {
    expect(readKind('embed')).toBe('embed')
    expect(readKind('crawl')).toBe('crawl')
    expect(readKind('Embed')).toBeUndefined()
    expect(readKind(null)).toBeUndefined()
  })
})

describe('countColumns', () => {
  it('names the units of one kind', () => {
    expect(countColumns('crawl').returned.label).toBe('Rows returned')
    expect(countColumns('crawl').credited.label).toBe('Rows counted')
    expect(countColumns('embed').returned.label).toBe('Texts returned')
    expect(countColumns('embed').credited.label).toBe('Chars counted')
    expect(countColumns('embed').checked.hint).toContain('0.999')
  })

  it('keeps labels neutral and explains both units when a table mixes kinds', () => {
    const mixed = countColumns(undefined)
    expect(mixed.returned.label).toBe('Returned')
    expect(mixed.credited.label).toBe('Counted')
    expect(mixed.credited.hint).toContain('Crawl: rows')
    expect(mixed.credited.hint).toContain('Embed: characters')
    expect(mixed.time.label).toBe('Work time')
  })
})

describe('formatTaskSize', () => {
  it('counts URLs for crawl and texts for embed', () => {
    expect(formatTaskSize('crawl', 1000, null)).toBe('1,000 URLs')
    expect(formatTaskSize('embed', 0, 2400)).toBe('2,400 texts')
    expect(formatTaskSize('embed', 0, null)).toBe('– texts')
  })
})

describe('withKind', () => {
  it('adds the kind only when it is not the default', () => {
    expect(withKind('/miners', 'crawl')).toBe('/miners')
    expect(withKind('/miners', 'embed')).toBe('/miners?kind=embed')
  })
})
