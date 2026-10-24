import { useState, type KeyboardEvent, type PointerEvent } from 'react'
import { EmptyState } from '@/components/ui/States'
import { useWidth } from '@/hooks/useWidth'
import { formatClock, formatCompact, formatInt, niceTicks, SECONDS_PER_HOUR } from '@/lib/format'

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
const TOOLTIP_OFFSET = 10
const RUNNING_OPACITY = 0.4

interface HourChartProps<P extends { at: number }> {
  label: string
  points: readonly P[]
  value: (point: P) => number
  /** The last point is the hour still running, drawn faded. */
  running?: boolean
  emptyTitle: string
}

function columnPath(x: number, top: number, bottom: number, width: number): string {
  const radius = Math.min(CAP, width / 2, bottom - top)
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
  value,
  running = false,
  emptyTitle,
}: HourChartProps<P>) {
  const [ref, width] = useWidth<HTMLDivElement>(FALLBACK_WIDTH)
  const [active, setActive] = useState<number | null>(null)
  const ticks = niceTicks(Math.max(0, ...points.map(value)))
  const top = ticks[ticks.length - 1] ?? 0
  const plotWidth = Math.max(width - PAD_LEFT - PAD_RIGHT, 1)
  const band = plotWidth / Math.max(points.length, 1)
  const columnWidth = Math.min(COLUMN_MAX, Math.max(band - GAP, 1))
  const xOf = (index: number) => PAD_LEFT + band * (index + 0.5)
  const yOf = (amount: number) =>
    PAD_TOP + PLOT_HEIGHT - (top > 0 ? (amount / top) * PLOT_HEIGHT : 0)
  const last = points.length - 1
  const isRunning = (index: number) => running && index === last

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
  return (
    <figure className="min-w-0">
      <figcaption className="text-ink mb-2 text-xs font-medium">{label}</figcaption>
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
            {points.map((point, index) =>
              value(point) > 0 ? (
                <path
                  key={point.at}
                  d={columnPath(
                    xOf(index) - columnWidth / 2,
                    yOf(value(point)),
                    yOf(0),
                    columnWidth,
                  )}
                  className="fill-series-1"
                  opacity={isRunning(index) ? RUNNING_OPACITY : undefined}
                />
              ) : null,
            )}
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
            className="border-line-strong bg-ground pointer-events-none absolute z-10 rounded-md border px-2.5 py-1.5 text-xs shadow-lg"
            style={
              xOf(active) > width / 2
                ? { right: width - xOf(active) + TOOLTIP_OFFSET, top: PAD_TOP }
                : { left: xOf(active) + TOOLTIP_OFFSET, top: PAD_TOP }
            }
          >
            <div className="text-ink-muted font-mono">
              {formatClock(shown.at)}–
              {isRunning(active) ? 'now, so far' : formatClock(shown.at + SECONDS_PER_HOUR)}
            </div>
            <div className="text-ink font-mono font-medium tabular-nums">
              {formatInt(value(shown))}
            </div>
          </div>
        ) : null}
      </div>
    </figure>
  )
}
