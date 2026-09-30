import type { MinerDetail } from '@/api/types'
import { Stat, StatGrid } from '@/components/ui/Stat'
import { VerdictCounts } from '@/components/ui/VerdictBadge'
import { formatInt, formatPercent, ratio } from '@/lib/format'
import { Eligibility } from './Eligibility'
import { Lockout } from './Lockout'

export function MinerStats({ miner }: { miner: MinerDetail }) {
  const { crawl } = miner.pools
  const { coverage, window } = miner
  return (
    <StatGrid>
      <Stat label="Share" value={formatPercent(miner.share)} detail="of the crawl pool" />
      <Stat
        label="Budget"
        value={formatInt(crawl.budget)}
        detail={`tasks at once · ${formatInt(crawl.in_flight)} held now`}
      />
      <Stat
        label="Coverage"
        value={formatPercent(coverage.coverage)}
        detail={
          coverage.assigned === undefined
            ? 'nothing finalized yet'
            : `${formatInt(coverage.returned)} rows for ${formatInt(coverage.assigned)} URLs`
        }
      />
      <Stat
        label="Eligible"
        value={<Eligibility coverage={coverage.coverage} eligible={coverage.eligible} />}
        detail="needs at least 85% coverage"
      />
      <Stat
        label={`Tasks · ${miner.window_hours} h`}
        value={formatInt(window.tasks)}
        detail={<VerdictCounts pass={window.pass} fail={window.fail} void={window.void} />}
      />
      <Stat
        label="Rows returned"
        value={formatInt(window.returned)}
        detail={`in the last ${miner.window_hours} h`}
      />
      <Stat
        label="Rows counted"
        value={formatInt(window.credited)}
        detail={`${formatPercent(ratio(window.credited, window.returned))} of rows returned`}
      />
      <Stat
        label="Lockout"
        value={<Lockout until={crawl.locked_until} />}
        detail="no new tasks while locked out"
      />
    </StatGrid>
  )
}
