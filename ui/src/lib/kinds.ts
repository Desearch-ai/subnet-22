import type { Kind } from '@/api/types'
import { formatInt } from './format'

export const KINDS: readonly Kind[] = ['crawl', 'embed']
export const DEFAULT_KIND: Kind = 'crawl'
export const KIND_PARAM = 'kind'

export function readKind(value: string | null): Kind | undefined {
  return KINDS.find((kind) => kind === value)
}

export const KIND_LABEL: Record<Kind, string> = {
  crawl: 'Crawl',
  embed: 'Embed',
}

export const KIND_SUMMARY: Record<Kind, string> = {
  crawl: 'up to 1,000 URLs a task, one row per URL',
  embed: 'page texts and passages, one vector per text',
}

interface KindWords {
  working: string
  returned: string
  credited: string
}

export const KIND_WORDS: Record<Kind, KindWords> = {
  crawl: { working: 'crawling', returned: 'rows', credited: 'rows' },
  embed: { working: 'embedding', returned: 'texts', credited: 'characters' },
}

export function formatTaskSize(kind: Kind, urls: number, texts: number | null): string {
  return kind === 'embed' ? `${formatInt(texts)} texts` : `${formatInt(urls)} URLs`
}

export interface ColumnText {
  label: string
  hint: string
}

export type CountColumn = 'returned' | 'missing' | 'checked' | 'credited' | 'time'

const UPLOAD_TIME_HINT = 'From the miner claiming the task to its upload being accepted'

const KIND_COLUMNS: Record<Kind, Record<CountColumn, ColumnText>> = {
  crawl: {
    returned: { label: 'Rows returned', hint: 'Rows the miner uploaded, one per URL' },
    missing: { label: 'Missing', hint: 'Assigned URLs with no row in the upload' },
    checked: { label: 'Checked', hint: 'Pages the validator fetched again: matched / checked' },
    credited: { label: 'Rows counted', hint: "Rows that count toward the miner's share" },
    time: { label: 'Crawl time', hint: UPLOAD_TIME_HINT },
  },
  embed: {
    returned: { label: 'Texts returned', hint: 'Texts the miner returned a vector for' },
    missing: { label: 'Missing', hint: 'Assigned texts with no vector in the upload' },
    checked: {
      label: 'Checked',
      hint: 'Texts the validator embedded again: matched / checked. Each needs a similarity of 0.999',
    },
    credited: {
      label: 'Chars counted',
      hint: "Characters embedded; all of them count toward the miner's share when the upload passes",
    },
    time: { label: 'Embed time', hint: UPLOAD_TIME_HINT },
  },
}

const ALL_KINDS_COLUMNS: Record<CountColumn, ColumnText> = {
  returned: {
    label: 'Returned',
    hint: 'Crawl: rows the miner uploaded. Embed: texts it returned a vector for',
  },
  missing: {
    label: 'Missing',
    hint: 'Crawl: assigned URLs with no row. Embed: assigned texts with no vector',
  },
  checked: {
    label: 'Checked',
    hint: 'Matched / checked. Crawl: pages the validator fetched again. Embed: texts it embedded again',
  },
  credited: {
    label: 'Counted',
    hint: "What counts toward the miner's share. Crawl: rows. Embed: characters",
  },
  time: { label: 'Work time', hint: UPLOAD_TIME_HINT },
}

/** Column texts for one kind, or worded for both when a table mixes them. */
export function countColumns(kind: Kind | undefined): Record<CountColumn, ColumnText> {
  return kind === undefined ? ALL_KINDS_COLUMNS : KIND_COLUMNS[kind]
}
