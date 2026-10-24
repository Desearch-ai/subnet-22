import type { ReactNode } from 'react'
import { useHours } from '@/api/queries/hours'
import type { HourPoint, Hours, Kind } from '@/api/types'
import { Card, CardHeader } from '@/components/ui/Card'
import { QueryState } from '@/components/ui/QueryState'
import { formatClock, formatInt } from '@/lib/format'
import { averagePerHour, splitHours } from '@/lib/work'
import { HourChart } from './HourChart'
import { HOUR_S, SERIES_COLORS, type HourSeries } from './hourSeries'

const DONE: HourSeries<HourPoint> = {
  key: 'tasks',
  label: 'Tasks done',
  color: SERIES_COLORS[0],
  value: (point) => point.tasks,
}

function Figure({ label, value, detail }: { label: string; value: ReactNode; detail: ReactNode }) {
  return (
    <div className="min-w-0">
      <div className="text-ink-muted text-xs">{label}</div>
      <div className="text-ink mt-0.5 text-2xl font-medium">{value}</div>
      <div className="text-ink-muted text-xs">{detail}</div>
    </div>
  )
}

function Summary({ hours }: { hours: Hours }) {
  const { done } = splitHours(hours.points)
  const last = done.at(-1)
  if (!last) return null
  const average = averagePerHour(done, (point) => point.tasks)
  return (
    <div className="grid gap-x-6 gap-y-4 sm:grid-cols-2">
      <Figure
        label="Done in the last hour"
        value={formatInt(last.tasks)}
        detail={`${formatClock(last.at)}–${formatClock(last.at + HOUR_S)}`}
      />
      <Figure
        label="Done in the last 24 hours"
        value={formatInt(done.reduce((sum, point) => sum + point.tasks, 0))}
        detail={`about ${formatInt(average)} an hour`}
      />
    </div>
  )
}

export function WorkPanel({ kind, miner }: { kind: Kind; miner?: string }) {
  const query = useHours(miner === undefined ? { kind } : { kind, miner })
  return (
    <Card>
      <CardHeader
        title={miner === undefined ? 'Tasks per hour' : 'Its tasks per hour'}
        description={`${kind === 'embed' ? 'Embed' : 'Crawl'} tasks done, last 24 hours`}
      />
      <QueryState query={query}>
        {(hours) => (
          <div className="space-y-6 p-4">
            <Summary hours={hours} />
            <HourChart
              label="Tasks done"
              points={hours.points}
              series={[DONE]}
              running
              emptyTitle={`No ${kind} task was done in the last 24 hours.`}
            />
          </div>
        )}
      </QueryState>
    </Card>
  )
}
