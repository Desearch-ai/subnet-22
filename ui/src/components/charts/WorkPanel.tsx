import { useState, type ReactNode } from 'react'
import { useHours } from '@/api/queries/hours'
import type { HourPoint, Hours, Kind } from '@/api/types'
import { Card, CardHeader } from '@/components/ui/Card'
import { QueryState } from '@/components/ui/QueryState'
import { SegmentedControl } from '@/components/ui/SegmentedControl'
import { formatInt } from '@/lib/format'
import { refusalText } from '@/lib/labels'
import {
  minutesInto,
  NO_CAPACITY,
  perMinute,
  QUEUE_EMPTY,
  refused,
  refusedOther,
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

const OTHER_COLOR = 'var(--color-ink-faint)'

function formatRate(value: number): string {
  return value >= 10 ? formatInt(value) : value.toFixed(1)
}

function issuedSeries(mine: boolean): HourSeries<HourPoint> {
  return {
    key: 'issued',
    label: mine ? 'Handed to it' : 'Handed out',
    color: SERIES_COLORS[0],
    value: (point) => point.issued,
  }
}

function finalizedSeries(mine: boolean): HourSeries<HourPoint> {
  return {
    key: 'tasks',
    label: mine ? 'Its verdicts' : 'Finalized',
    color: SERIES_COLORS[1],
    value: (point) => point.tasks,
  }
}

/** For the subnet, miners turned away each minute; for one miner, the minutes it was. */
function refusalSeries(mine: boolean, hours: Hours): HourSeries<HourPoint>[] {
  const running = hours.points.at(-1)
  const minutes = (point: HourPoint) =>
    point === running ? minutesInto(point, hours.as_of) : undefined
  return [
    {
      key: QUEUE_EMPTY,
      label: refusalText(QUEUE_EMPTY).label,
      color: SERIES_COLORS[2],
      value: (point) =>
        mine ? refused(point, QUEUE_EMPTY) : perMinute(refused(point, QUEUE_EMPTY), minutes(point)),
    },
    {
      key: NO_CAPACITY,
      label: refusalText(NO_CAPACITY).label,
      color: SERIES_COLORS[1],
      value: (point) =>
        mine ? refused(point, NO_CAPACITY) : perMinute(refused(point, NO_CAPACITY), minutes(point)),
    },
    {
      key: 'other',
      label: 'Other reasons',
      color: OTHER_COLOR,
      value: (point) =>
        mine ? refusedOther(point) : perMinute(refusedOther(point), minutes(point)),
    },
  ]
}

function Figure({ label, value, detail }: { label: string; value: ReactNode; detail: ReactNode }) {
  return (
    <div className="min-w-0">
      <div className="text-ink-faint text-label font-mono tracking-wider uppercase">{label}</div>
      <div className="text-ink mt-0.5 text-lg font-medium">{value}</div>
      <div className="text-ink-muted text-xs">{detail}</div>
    </div>
  )
}

function Summary({ hours, mine }: { hours: Hours; mine: boolean }) {
  const { done, current } = splitHours(hours.points)
  const last = done.at(-1)
  if (!last || !current) return null
  const minutes = minutesInto(current, hours.as_of)
  const turnedAway = (point: HourPoint, code: string, over?: number) =>
    mine ? formatInt(refused(point, code)) : formatRate(perMinute(refused(point, code), over))
  const refusedWord = mine ? 'minutes refused' : 'miners a minute'
  return (
    <div className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-4">
      <Figure
        label={mine ? 'Handed to it · last hour' : 'Handed out · last hour'}
        value={formatInt(last.issued)}
        detail={`${formatInt(current.issued)} so far this hour`}
      />
      <Figure
        label={mine ? 'Its verdicts · last hour' : 'Finalized · last hour'}
        value={formatInt(last.tasks)}
        detail={`${formatInt(last.pass)} passed, ${formatInt(last.fail)} failed, ${formatInt(last.void)} void`}
      />
      <Figure
        label={`Queue empty · last hour`}
        value={turnedAway(last, QUEUE_EMPTY)}
        detail={`${refusedWord} · ${turnedAway(current, QUEUE_EMPTY, minutes)} this hour`}
      />
      <Figure
        label={`No capacity · last hour`}
        value={turnedAway(last, NO_CAPACITY)}
        detail={`${refusedWord} · ${turnedAway(current, NO_CAPACITY, minutes)} this hour`}
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
  const issued = issuedSeries(mine)
  const finalized = finalizedSeries(mine)
  return (
    <Card>
      <CardHeader
        title={mine ? 'Its work per hour' : 'Is there work?'}
        description={
          mine
            ? `Its ${kind} tasks each hour of the last 24, and the minutes its claims were turned away`
            : `${kind === 'embed' ? 'Embed' : 'Crawl'} tasks handed out and finalized each hour of the last 24, and how many miners asked and got nothing`
        }
        actions={
          <>
            {actions}
            <SegmentedControl label="Show as" segments={VIEWS} value={view} onChange={setView} />
          </>
        }
      />
      <QueryState query={query}>
        {(hours) => {
          const refusals = refusalSeries(mine, hours)
          return (
            <div className="space-y-5 p-4">
              <Summary hours={hours} mine={mine} />
              {view === 'chart' ? (
                <div className="grid gap-5 xl:grid-cols-2">
                  <HourChart
                    label="Tasks per hour"
                    points={hours.points}
                    series={[issued]}
                    line={finalized}
                    running
                    emptyTitle={`No ${kind} task was handed out in the last 24 hours.`}
                  />
                  <HourChart
                    label={
                      mine ? 'Minutes its claims were refused' : 'Miners turned away, per minute'
                    }
                    points={hours.points}
                    series={refusals}
                    running
                    format={mine ? formatInt : formatRate}
                    emptyTitle="No claim was refused in the last 24 hours."
                  />
                </div>
              ) : (
                <HourTable
                  points={hours.points}
                  running
                  columns={[
                    { series: [issued, finalized] },
                    { series: refusals, format: mine ? formatInt : formatRate },
                  ]}
                />
              )}
              <p className="text-ink-faint text-xs">
                {mine
                  ? 'A refused claim is counted once a minute per reason, so an hour holds at most 60 minutes of each.'
                  : 'A refused claim is counted once a minute per miner and reason, so an hour’s refusals over 60 is how many miners were turned away each minute.'}
              </p>
            </div>
          )
        }}
      </QueryState>
    </Card>
  )
}
