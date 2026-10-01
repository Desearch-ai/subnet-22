import type { Overview } from '@/api/types'
import { Stat, StatGrid } from '@/components/ui/Stat'
import { formatDuration, formatInt, formatPercent, ratio } from '@/lib/format'

export function OverviewStats({ overview }: { overview: Overview }) {
  const { total, validators } = overview
  const finalized = total.pass + total.fail + total.void
  return (
    <StatGrid>
      <Stat
        label="Tasks waiting"
        value={formatInt(overview.queue.crawl)}
        detail="ready for a miner to claim"
      />
      <Stat
        label="Being checked"
        value={formatInt(overview.validating)}
        detail={
          overview.oldest_validation_s === null
            ? 'uploads waiting for votes'
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
        detail={`with a finalized task in ${overview.window_hours} h`}
      />
      <Stat
        label="Pass rate"
        value={formatPercent(ratio(total.pass, finalized))}
        detail={`of ${formatInt(finalized)} finalized tasks`}
      />
    </StatGrid>
  )
}
