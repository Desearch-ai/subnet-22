import { Fragment, type ReactNode } from 'react'
import type { Hours, Kind, Overview } from '@/api/types'
import { Stat, StatGrid } from '@/components/ui/Stat'
import { formatDuration, formatInt, formatPercent } from '@/lib/format'
import { KINDS } from '@/lib/kinds'
import { queueMinutes, splitHours } from '@/lib/work'

/** A figure for both kinds takes the whole row where tiles sit two abreast. */
const BOTH_KINDS = 'col-span-2 md:col-span-1'

function ByKind({ value }: { value: (kind: Kind) => ReactNode }) {
  return KINDS.map((kind, index) => (
    <Fragment key={kind}>
      {index > 0 ? <span className="text-ink-faint"> · </span> : null}
      {value(kind)}
      <span className="text-ink-muted text-xs font-normal"> {kind}</span>
    </Fragment>
  ))
}

function queueDetail(kind: Kind, waiting: number, hours: Hours | undefined): string {
  const lastHour = hours ? splitHours(hours.points).done.at(-1) : undefined
  const minutes = lastHour ? queueMinutes(waiting, lastHour.tasks) : null
  if (minutes === null) return 'ready for a miner to claim'
  const lasts = minutes < 1 ? 'less than a minute' : `about ${formatDuration(minutes * 60)}`
  return `${kind}: ${lasts} of work at the last hour’s pace`
}

interface LiveStatsProps {
  overviews: Record<Kind, Overview>
  kind: Kind
  hours: Hours | undefined
}

export function LiveStats({ overviews, kind, hours }: LiveStatsProps) {
  const { crawl } = overviews
  const chosen = overviews[kind]
  return (
    <StatGrid>
      <Stat
        className={BOTH_KINDS}
        label="Tasks waiting"
        value={<ByKind value={(each) => formatInt(crawl.queue[each])} />}
        detail={queueDetail(kind, crawl.queue[kind], hours)}
      />
      <Stat
        label="Claimed now"
        value={formatInt(crawl.claimed)}
        detail="held by miners, not uploaded yet"
      />
      <Stat
        label="Being checked"
        value={formatInt(crawl.validating)}
        detail={
          crawl.oldest_validation_s === null
            ? 'uploads waiting for their verdict'
            : `oldest waiting ${formatDuration(crawl.oldest_validation_s)}`
        }
      />
      <Stat
        label="Validators online"
        value={`${formatInt(crawl.validators.active)} of ${formatInt(crawl.validators.known)}`}
        detail={
          crawl.validators.agreement === null
            ? 'no checked upload to agree on yet'
            : `agree with ${formatPercent(crawl.validators.agreement)} of final results`
        }
      />
      <Stat
        className={BOTH_KINDS}
        label="Miners"
        value={<ByKind value={(each) => formatInt(overviews[each].miners)} />}
        detail={
          chosen.locked_out > 0
            ? `${formatInt(chosen.locked_out)} locked out of ${kind} now`
            : `with a finalized task in ${crawl.window_hours} h`
        }
      />
    </StatGrid>
  )
}
