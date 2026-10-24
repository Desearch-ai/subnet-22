import { useState, type ReactNode } from 'react'
import { useHours } from '@/api/queries/hours'
import type { HourPoint, Hours, Kind } from '@/api/types'
import { Card, CardHeader } from '@/components/ui/Card'
import { QueryState } from '@/components/ui/QueryState'
import { SegmentedControl } from '@/components/ui/SegmentedControl'
import { formatInt } from '@/lib/format'
import {
  averagePerHour,
  compareToAverage,
  minutesInto,
  perMinute,
  QUEUE_EMPTY,
  refused,
  splitHours,
} from '@/lib/work'
import { HourChart } from './HourChart'
import { HourTable } from './HourTable'
import { SERIES_COLORS, type HourSeries } from './hourSeries'

type View = 'chart' | 'table'

const VIEWS = [
  { value: 'chart', label: 'Chart' },
  { value: 'table', label: 'Table' },
] as const

function formatRate(value: number): string {
  return value >= 10 ? formatInt(value) : value.toFixed(1)
}

const DONE: HourSeries<HourPoint> = {
  key: 'tasks',
  label: 'Tasks done',
  color: SERIES_COLORS[0],
  value: (point) => point.tasks,
}

/** For the subnet, miners that found no task each minute; for one miner, the minutes it did. */
function noTaskSeries(mine: boolean, hours: Hours): HourSeries<HourPoint> {
  const running = hours.points.at(-1)
  return {
    key: QUEUE_EMPTY,
    label: mine ? 'Minutes with no task' : 'Miners with no task, per minute',
    color: SERIES_COLORS[2],
    value: (point) =>
      mine
        ? refused(point, QUEUE_EMPTY)
        : perMinute(
            refused(point, QUEUE_EMPTY),
            point === running ? minutesInto(point, hours.as_of) : undefined,
          ),
  }
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

function Summary({ hours, mine }: { hours: Hours; mine: boolean }) {
  const { done } = splitHours(hours.points)
  const last = done.at(-1)
  if (!last) return null
  const average = averagePerHour(done, (point) => point.tasks)
  const noTask = refused(last, QUEUE_EMPTY)
  return (
    <div className="grid gap-x-6 gap-y-4 sm:grid-cols-3">
      <Figure
        label="Done in the last hour"
        value={formatInt(last.tasks)}
        detail={compareToAverage(last.tasks, average)}
      />
      <Figure
        label="Done in the last 24 hours"
        value={formatInt(done.reduce((sum, point) => sum + point.tasks, 0))}
        detail={`about ${formatInt(average)} an hour`}
      />
      <Figure
        label={mine ? 'Minutes it found no task, last hour' : 'Miners with no task, last hour'}
        value={mine ? `${formatInt(noTask)} of 60` : `${formatRate(perMinute(noTask))} a minute`}
        detail="asked for a task when none was waiting"
      />
    </div>
  )
}

interface WorkPanelProps {
  kind: Kind
  miner?: string
  actions?: ReactNode
}

export function WorkPanel({ kind, miner, actions }: WorkPanelProps) {
  const query = useHours(miner === undefined ? { kind } : { kind, miner })
  const [view, setView] = useState<View>('chart')
  const mine = miner !== undefined
  return (
    <Card>
      <CardHeader
        title={mine ? 'Its tasks per hour' : 'Tasks per hour'}
        description={`${kind === 'embed' ? 'Embed' : 'Crawl'} tasks done, last 24 hours`}
        actions={
          <>
            {actions}
            <SegmentedControl label="Show as" segments={VIEWS} value={view} onChange={setView} />
          </>
        }
      />
      <QueryState query={query}>
        {(hours) => {
          const noTask = noTaskSeries(mine, hours)
          return (
            <div className="space-y-6 p-4">
              <Summary hours={hours} mine={mine} />
              {view === 'chart' ? (
                <div className="grid gap-6 xl:grid-cols-2">
                  <HourChart
                    label="Tasks done"
                    points={hours.points}
                    series={[DONE]}
                    running
                    emptyTitle={`No ${kind} task was done in the last 24 hours.`}
                  />
                  <HourChart
                    label={noTask.label}
                    points={hours.points}
                    series={[noTask]}
                    running
                    format={mine ? formatInt : formatRate}
                    emptyTitle="Every miner that asked got a task."
                  />
                </div>
              ) : (
                <HourTable
                  points={hours.points}
                  running
                  columns={[
                    { series: [DONE] },
                    { series: [noTask], format: mine ? formatInt : formatRate },
                  ]}
                />
              )}
            </div>
          )
        }}
      </QueryState>
    </Card>
  )
}
