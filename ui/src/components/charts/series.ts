import type { Kind, TaskSeriesPoint } from '@/api/types'

export interface SeriesDef {
  key: 'tasks' | 'pass' | 'returned' | 'credited'
  label: string
  color: string
}

interface KindSeries {
  title: string
  caption: string
  series: readonly SeriesDef[]
  emptyTitle: string
}

const FIRST_COLOR = 'var(--color-series-returned)'
const SECOND_COLOR = 'var(--color-series-counted)'

// Embed credit is all or nothing per task, so tasks passed is its counterpart of rows counted.
export const KIND_SERIES: Record<Kind, KindSeries> = {
  crawl: {
    title: 'Rows over time',
    caption: 'Rows returned and counted',
    series: [
      { key: 'returned', label: 'Rows returned', color: FIRST_COLOR },
      { key: 'credited', label: 'Rows counted', color: SECOND_COLOR },
    ],
    emptyTitle: 'No rows were returned in this period.',
  },
  embed: {
    title: 'Embed tasks over time',
    caption: 'Embed tasks finalized and passed',
    series: [
      { key: 'tasks', label: 'Tasks finalized', color: FIRST_COLOR },
      { key: 'pass', label: 'Tasks passed', color: SECOND_COLOR },
    ],
    emptyTitle: 'No embed task was finalized in this period.',
  },
}

export type SeriesPoint = TaskSeriesPoint
