import { useState, type KeyboardEvent, type PointerEvent } from 'react'
import { EmptyState } from '@/components/ui/States'
import { useWidth } from '@/hooks/useWidth'
import { formatClock, formatCompact, formatInt, niceTicks } from '@/lib/format'
import { ChartLegend, SeriesKey, type LegendItem } from './ChartLegend'
import { HOUR_S, type HourSeries } from './hourSeries'

const HEIGHT = 184
const PAD_TOP = 10
const PAD_RIGHT = 12
const PAD_BOTTOM = 24
const PAD_LEFT = 44
const PLOT_HEIGHT = HEIGHT - PAD_TOP - PAD_BOTTOM
const FALLBACK_WIDTH = 800
const LABEL_EVERY = 4
const COLUMN_MAX = 24
const GAP = 2
const CAP = 4
const DOT = 4
const TOOLTIP_OFFSET = 10
const RUNNING_OPACITY = 0.4

interface HourChartProps<P extends { at: number }> {
  label: string
  points: readonly P[]
  /** Stacked as columns, the first at the bottom. */
  series: readonly HourSeries<P>[]
  /** Drawn as a line over the columns, through the hours that are over. */
  line?: HourSeries<P>
  /** The last point is the hour still running, drawn faded. */
  running?: boolean
  format?: (value: number) => string
  emptyTitle: string
}

function columnPath(
  x: number,
  top: number,
  bottom: number,
  width: number,
  rounded: boolean,
): string {
  const radius = rounded ? Math.min(CAP, width / 2, bottom - top) : 0
  return [
    `M${x},${bottom}`,
    `V${top + radius}`,
    `Q${x},${top} ${x + radius},${top}`,
    `H${x + width - radius}`,
    `Q${x + width},${top} ${x + width},${top + radius}`,
    `V${bottom}Z`,
  ].join(' ')
}

export function HourChart<P extends { at: number }>({
  label,
  points,
  series,
  line,
  running = false,
  format = formatInt,
  emptyTitle,
}: HourChartProps<P>) {
  const [ref, width] = useWidth<HTMLDivElement>(FALLBACK_WIDTH)
  const [active, setActive] = useState<number | null>(null)
  const heights = points.map((point) =>
    Math.max(
      series.reduce((sum, item) => sum + item.value(point), 0),
      line ? line.value(point) : 0,
    ),
  )
  const ticks = niceTicks(Math.max(0, ...heights))
  const top = ticks[ticks.length - 1] ?? 0
  const plotWidth = Math.max(width - PAD_LEFT - PAD_RIGHT, 1)
  const band = plotWidth / Math.max(points.length, 1)
  const columnWidth = Math.min(COLUMN_MAX, Math.max(band - GAP, 1))
  const xOf = (index: number) => PAD_LEFT + band * (index + 0.5)
  const yOf = (value: number) => PAD_TOP + PLOT_HEIGHT - (top > 0 ? (value / top) * PLOT_HEIGHT : 0)
  const last = points.length - 1
  const isRunning = (index: number) => running && index === last
  const lineEnd = running ? last - 1 : last
  const legend: (HourSeries<P> & LegendItem)[] = [
    ...series.map((item) => ({ ...item, mark: 'column' as const })),
    ...(line ? [{ ...line, mark: 'line' as const }] : []),
  ]

  const pick = (event: PointerEvent<SVGRectElement>) => {
    const box = event.currentTarget.getBoundingClientRect()
    const index = Math.floor(((event.clientX - box.left) / box.width) * points.length)
    setActive(Math.min(last, Math.max(0, index)))
  }
  const step = (event: KeyboardEvent<SVGSVGElement>) => {
    const moves: Record<string, number> = { ArrowLeft: -1, ArrowRight: 1 }
    if (event.key in moves) {
      event.preventDefault()
      setActive((index) => Math.min(last, Math.max(0, (index ?? last) + (moves[event.key] ?? 0))))
    } else if (event.key === 'Escape') {
      setActive(null)
    }
  }

  const shown = active === null ? undefined : points[active]
  const dotAt = active ?? (lineEnd >= 0 ? lineEnd : null)
  const dotPoint = dotAt === null ? undefined : points[dotAt]
  return (
    <figure className="min-w-0">
      <figcaption className="mb-2 flex flex-wrap items-center justify-between gap-x-4 gap-y-1">
        <span className="text-ink text-xs font-medium">{label}</span>
        {legend.length > 1 ? <ChartLegend items={legend} /> : null}
      </figcaption>
      <div ref={ref} className="relative">
        {top === 0 ? (
          <EmptyState title={emptyTitle} />
        ) : (
          <svg
            width={width}
            height={HEIGHT}
            role="img"
            aria-label={`${label}. Use the arrow keys to read each hour.`}
            tabIndex={0}
            onKeyDown={step}
            onFocus={() => {
              setActive(last)
            }}
            onBlur={() => {
              setActive(null)
            }}
            className="fill-ink-faint block font-mono text-[11px] outline-none"
          >
            {ticks.map((tick) => (
              <g key={tick}>
                <line
                  x1={PAD_LEFT}
                  x2={width - PAD_RIGHT}
                  y1={yOf(tick)}
                  y2={yOf(tick)}
                  className="stroke-line"
                  shapeRendering="crispEdges"
                />
                <text x={PAD_LEFT - 8} y={yOf(tick) + 4} textAnchor="end">
                  {formatCompact(tick)}
                </text>
              </g>
            ))}
            {points.map((point, index) =>
              (last - index) % LABEL_EVERY === 0 ? (
                <text key={point.at} x={xOf(index)} y={HEIGHT - 6} textAnchor="middle">
                  {isRunning(index) ? 'now' : formatClock(point.at)}
                </text>
              ) : null,
            )}
            {active !== null ? (
              <rect
                x={xOf(active) - band / 2}
                y={PAD_TOP}
                width={band}
                height={PLOT_HEIGHT}
                className="fill-raised"
              />
            ) : null}
            {points.map((point, index) => {
              let below = 0
              const filled = series.filter((item) => item.value(point) > 0)
              return (
                <g key={point.at} opacity={isRunning(index) ? RUNNING_OPACITY : undefined}>
                  {filled.map((item, position) => {
                    const bottom = yOf(below) - (position > 0 ? GAP : 0)
                    below += item.value(point)
                    const segmentTop = yOf(below)
                    if (bottom - segmentTop < 0.5) return null
                    return (
                      <path
                        key={item.key}
                        d={columnPath(
                          xOf(index) - columnWidth / 2,
                          segmentTop,
                          bottom,
                          columnWidth,
                          position === filled.length - 1,
                        )}
                        fill={item.color}
                      />
                    )
                  })}
                </g>
              )
            })}
            {line && lineEnd > 0 ? (
              <polyline
                fill="none"
                stroke={line.color}
                strokeWidth={2}
                strokeLinejoin="round"
                strokeLinecap="round"
                points={points
                  .slice(0, lineEnd + 1)
                  .map((point, index) => `${xOf(index)},${yOf(line.value(point))}`)
                  .join(' ')}
              />
            ) : null}
            {line && dotAt !== null && dotPoint ? (
              <circle
                cx={xOf(dotAt)}
                cy={yOf(line.value(dotPoint))}
                r={DOT}
                fill={isRunning(dotAt) ? 'var(--color-surface)' : line.color}
                stroke={isRunning(dotAt) ? line.color : 'var(--color-surface)'}
                strokeWidth={2}
              />
            ) : null}
            <rect
              x={PAD_LEFT}
              y={0}
              width={plotWidth}
              height={HEIGHT}
              fill="transparent"
              onPointerMove={pick}
              onPointerDown={pick}
              onPointerLeave={() => {
                setActive(null)
              }}
            />
          </svg>
        )}
        {shown !== undefined && active !== null && top > 0 ? (
          <div
            role="status"
            className="border-line-strong bg-ground pointer-events-none absolute z-10 min-w-36 rounded-md border px-2.5 py-1.5 text-xs shadow-lg"
            style={
              xOf(active) > width / 2
                ? { right: width - xOf(active) + TOOLTIP_OFFSET, top: PAD_TOP }
                : { left: xOf(active) + TOOLTIP_OFFSET, top: PAD_TOP }
            }
          >
            <div className="text-ink-muted mb-1 font-mono">
              {formatClock(shown.at)}–
              {isRunning(active) ? 'now, so far' : formatClock(shown.at + HOUR_S)}
            </div>
            <ul className="space-y-0.5">
              {legend.map((item) => (
                <li key={item.key} className="flex items-center gap-2 whitespace-nowrap">
                  <SeriesKey color={item.color} mark="line" />
                  <span className="text-ink font-mono font-medium tabular-nums">
                    {format(item.value(shown))}
                  </span>
                  <span className="text-ink-muted">{item.label}</span>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </div>
    </figure>
  )
}
