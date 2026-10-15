import { Fragment } from 'react'
import type { Overview } from '@/api/types'
import { Stat, StatGrid } from '@/components/ui/Stat'
import { formatDuration, formatInt, formatPercent, ratio } from '@/lib/format'
import { KINDS } from '@/lib/kinds'

function QueueCounts({ queue }: { queue: Overview['queue'] }) {
  return KINDS.map((kind, index) => (
    <Fragment key={kind}>
      {index > 0 ? <span className="text-ink-faint"> · </span> : null}
      {formatInt(queue[kind])}
      <span className="text-ink-muted text-xs font-normal"> {kind}</span>
    </Fragment>
  ))
}

export function OverviewStats({ overview }: { overview: Overview }) {
  const { kind, validators, window } = overview
  const finalized = window.pass + window.fail + window.void
  return (
    <StatGrid>
      <Stat
        label="Tasks waiting"
        value={<QueueCounts queue={overview.queue} />}
        detail="ready for a miner to claim"
      />
      <Stat
        label="Being checked"
        value={formatInt(overview.validating)}
        detail={
          overview.oldest_validation_s === null
            ? 'uploads of both kinds waiting for votes'
            : `oldest waiting ${formatDuration(overview.oldest_validation_s)}`
        }
      />
      <Stat
        label="Active validators"
        value={formatInt(validators.active)}
        detail={`of ${formatInt(validators.known)} seen`}
      />
      <Stat
        label="Miners"
        value={formatInt(overview.miners)}
        detail={`with a finalized ${kind} task in ${overview.window_hours} h`}
      />
      <Stat
        label={`Pass rate · ${overview.window_hours} h`}
        value={formatPercent(ratio(window.pass, finalized))}
        detail={`of ${formatInt(finalized)} finalized ${kind} tasks`}
      />
    </StatGrid>
  )
}
