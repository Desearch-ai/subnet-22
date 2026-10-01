import { EmptyState } from '@/components/ui/States'
import { formatClock, formatInt } from '@/lib/format'
import { SeriesLegend } from './SeriesLegend'
import type { SeriesDef, SeriesPoint } from './series'

const WIDTH = 1000
const HEIGHT = 200
const PAD_LEFT = 48
const PAD_RIGHT = 8
const PAD_TOP = 8
const PAD_BOTTOM = 22
const PLOT_WIDTH = WIDTH - PAD_LEFT - PAD_RIGHT
const PLOT_HEIGHT = HEIGHT - PAD_TOP - PAD_BOTTOM
const Y_STEPS = 4
const X_LABEL_EVERY = 4

interface SeriesChartProps {
  title: string
  points: readonly SeriesPoint[]
  series: readonly SeriesDef[]
  emptyTitle: string
}

export function SeriesChart({ title, points, series, emptyTitle }: SeriesChartProps) {
  const highest = Math.max(0, ...points.flatMap((point) => series.map((item) => point[item.key])))
  const xOf = (index: number) =>
    PAD_LEFT + (points.length > 1 ? (index / (points.length - 1)) * PLOT_WIDTH : PLOT_WIDTH / 2)
  const yOf = (value: number) => PAD_TOP + PLOT_HEIGHT - (value / highest) * PLOT_HEIGHT

  return (
    <figure className="min-w-0">
      <figcaption className="mb-2 flex flex-wrap items-center justify-between gap-x-4 gap-y-1">
        <span className="text-ink text-xs font-medium">{title}</span>
        <SeriesLegend series={series} />
      </figcaption>
      {highest === 0 ? (
        <EmptyState title={emptyTitle} />
      ) : (
        <svg
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          role="img"
          aria-label={title}
          className="fill-ink-faint w-full font-mono text-[11px]"
        >
          {Array.from({ length: Y_STEPS + 1 }, (_, step) => {
            const value = (highest / Y_STEPS) * step
            return (
              <g key={step}>
                <line
                  x1={PAD_LEFT}
                  x2={WIDTH - PAD_RIGHT}
                  y1={yOf(value)}
                  y2={yOf(value)}
                  className="stroke-line"
                />
                <text x={PAD_LEFT - 8} y={yOf(value) + 4} textAnchor="end">
                  {formatInt(Math.round(value))}
                </text>
              </g>
            )
          })}
          {points.map((point, index) =>
            index % X_LABEL_EVERY === 0 ? (
              <text key={point.at} x={xOf(index)} y={HEIGHT - 4} textAnchor="middle">
                {formatClock(point.at)}
              </text>
            ) : null,
          )}
          {series.map((item) => (
            <g key={item.key} stroke={item.color} fill={item.color}>
              <polyline
                fill="none"
                strokeWidth={2}
                strokeLinejoin="round"
                points={points
                  .map((point, index) => `${xOf(index)},${yOf(point[item.key])}`)
                  .join(' ')}
              />
              {points.map((point, index) =>
                point[item.key] > 0 ? (
                  <circle key={point.at} cx={xOf(index)} cy={yOf(point[item.key])} r={3}>
                    <title>
                      {`${formatClock(point.at)} · ${item.label}: ${formatInt(point[item.key])}`}
                    </title>
                  </circle>
                ) : null,
              )}
            </g>
          ))}
        </svg>
      )}
    </figure>
  )
}
