import type { SeriesDef } from './series'

export function SeriesLegend({ series }: { series: readonly SeriesDef[] }) {
  return (
    <ul className="text-ink-muted flex flex-wrap gap-x-4 gap-y-1 text-xs">
      {series.map((item) => (
        <li key={item.key} className="inline-flex items-center gap-1.5">
          <span
            aria-hidden
            className="h-0.5 w-3 rounded-full"
            style={{ backgroundColor: item.color }}
          />
          {item.label}
        </li>
      ))}
    </ul>
  )
}
