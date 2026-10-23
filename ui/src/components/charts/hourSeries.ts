export interface HourSeries<P> {
  key: string
  label: string
  color: string
  value: (point: P) => number
}

export const SERIES_COLORS = [
  'var(--color-series-1)',
  'var(--color-series-2)',
  'var(--color-series-3)',
] as const

export const HOUR_S = 3600
