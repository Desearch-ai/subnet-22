import { cn } from '@/lib/cn'

export type Mark = 'column' | 'line'

export interface LegendItem {
  key: string
  label: string
  color: string
  mark: Mark
}

export function SeriesKey({ color, mark }: { color: string; mark: Mark }) {
  return (
    <span
      aria-hidden
      className={cn('shrink-0', mark === 'line' ? 'h-0.5 w-3 rounded-full' : 'size-2.5 rounded-sm')}
      style={{ backgroundColor: color }}
    />
  )
}

export function ChartLegend({ items }: { items: readonly LegendItem[] }) {
  return (
    <ul className="text-ink-muted flex flex-wrap gap-x-4 gap-y-1 text-xs">
      {items.map((item) => (
        <li key={item.key} className="inline-flex items-center gap-1.5">
          <SeriesKey color={item.color} mark={item.mark} />
          {item.label}
        </li>
      ))}
    </ul>
  )
}
