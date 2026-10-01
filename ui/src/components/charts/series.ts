import type { TaskSeriesPoint } from '@/api/types'

export interface SeriesDef {
  key: 'returned' | 'credited'
  label: string
  color: string
}

export const ROW_SERIES: readonly SeriesDef[] = [
  { key: 'returned', label: 'Rows returned', color: 'var(--color-series-returned)' },
  { key: 'credited', label: 'Rows counted', color: 'var(--color-series-counted)' },
]

export type SeriesPoint = TaskSeriesPoint
