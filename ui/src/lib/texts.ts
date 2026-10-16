import type { TaskText } from '@/api/types'
import { compareValues } from './sort'

export const EMBED_SAMPLES = 48
export const MATCH_SIMILARITY = 0.999

export function textName(text: Pick<TaskText, 'kind' | 'index'>): string {
  return text.kind === 'page' ? 'Page text' : `Passage ${text.index}`
}

export function isBelowBar(similarity: number | null): boolean {
  return similarity !== null && similarity < MATCH_SIMILARITY
}

export function sortTexts(texts: readonly TaskText[]): TaskText[] {
  const failed = (text: TaskText) => Number(text.outcome === 'mismatched')
  return texts.toSorted(
    (one, other) =>
      failed(other) - failed(one) ||
      compareValues(one.similarity, other.similarity) ||
      compareValues(one.text_id, other.text_id),
  )
}

export function lowestSimilarity(texts: readonly TaskText[]): number | null {
  const known = texts.flatMap((text) => (text.similarity === null ? [] : [text.similarity]))
  return known.length === 0 ? null : Math.min(...known)
}
