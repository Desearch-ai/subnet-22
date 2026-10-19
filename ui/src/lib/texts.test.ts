import { describe, expect, it } from 'vitest'
import type { TaskText } from '@/api/types'
import { isBelowBar, lowestSimilarity, sortTexts, textName } from './texts'

function text(text_id: string, outcome: TaskText['outcome'], similarity: number | null): TaskText {
  return { text_id, kind: 'page', index: 0, url: null, outcome, similarity }
}

describe('sortTexts', () => {
  it('puts mismatches first, then the lowest similarity, without changing the input', () => {
    const texts = [
      text('a', 'matched', 0.99995),
      text('b', 'mismatched', 0.9981),
      text('c', 'matched', 0.99912),
      text('d', 'unverifiable', null),
      text('e', 'mismatched', 0.42),
    ]
    expect(sortTexts(texts).map((row) => row.text_id)).toEqual(['e', 'b', 'd', 'c', 'a'])
    expect(texts.map((row) => row.text_id)).toEqual(['a', 'b', 'c', 'd', 'e'])
  })

  it('orders equal similarities by text id', () => {
    const texts = [text('y', 'matched', 0.9995), text('x', 'matched', 0.9995)]
    expect(sortTexts(texts).map((row) => row.text_id)).toEqual(['x', 'y'])
  })
})

describe('lowestSimilarity', () => {
  it('ignores texts without a similarity', () => {
    expect(lowestSimilarity([text('a', 'matched', 0.9996), text('b', 'unverifiable', null)])).toBe(
      0.9996,
    )
    expect(lowestSimilarity([text('b', 'unverifiable', null)])).toBeNull()
    expect(lowestSimilarity([])).toBeNull()
  })
})

describe('textName', () => {
  it('names the page text and numbers passages', () => {
    expect(textName({ kind: 'page', index: 0 })).toBe('Page text')
    expect(textName({ kind: 'passage', index: 3 })).toBe('Passage 3')
  })
})

describe('isBelowBar', () => {
  it('fails only a known similarity under 0.999', () => {
    expect(isBelowBar(0.998999)).toBe(true)
    expect(isBelowBar(0.999)).toBe(false)
    expect(isBelowBar(null)).toBe(false)
  })
})
