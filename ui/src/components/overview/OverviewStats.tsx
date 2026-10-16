import { Fragment, type ReactNode } from 'react'
import type { Kind, Overview } from '@/api/types'
import { Stat, StatGrid } from '@/components/ui/Stat'
import { formatDuration, formatInt, formatPercent, ratio } from '@/lib/format'
import { KINDS } from '@/lib/kinds'

function ByKind({ value }: { value: (kind: Kind) => ReactNode }) {
  return KINDS.map((kind, index) => (
    <Fragment key={kind}>
      {index > 0 ? <span className="text-ink-faint"> · </span> : null}
      {value(kind)}
      <span className="text-ink-muted text-xs font-normal"> {kind}</span>
    </Fragment>
  ))
}

function finalized({ window }: Overview): number {
  return window.pass + window.fail + window.void
}

export function OverviewStats({ overviews }: { overviews: Record<Kind, Overview> }) {
  const { crawl } = overviews
  return (
    <StatGrid>
      <Stat
        label="Tasks waiting"
        value={<ByKind value={(kind) => formatInt(crawl.queue[kind])} />}
        detail="ready for a miner to claim"
      />
      <Stat
        label="Being checked"
        value={formatInt(crawl.validating)}
        detail={
          crawl.oldest_validation_s === null
            ? 'uploads of both kinds waiting for votes'
            : `oldest waiting ${formatDuration(crawl.oldest_validation_s)}`
        }
      />
      <Stat
        label="Active validators"
        value={formatInt(crawl.validators.active)}
        detail={`of ${formatInt(crawl.validators.known)} seen`}
      />
      <Stat
        label="Miners"
        value={<ByKind value={(kind) => formatInt(overviews[kind].miners)} />}
        detail={`with a finalized task in ${crawl.window_hours} h`}
      />
      <Stat
        label={`Pass rate · ${crawl.window_hours} h`}
        value={
          <ByKind
            value={(kind) =>
              formatPercent(ratio(overviews[kind].window.pass, finalized(overviews[kind])))
            }
          />
        }
        detail={
          <>
            of <ByKind value={(kind) => formatInt(finalized(overviews[kind]))} /> finalized tasks
          </>
        }
      />
    </StatGrid>
  )
}
