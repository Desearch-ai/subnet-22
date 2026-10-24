import type { Kind, MinerDetail } from '@/api/types'
import { Stat } from '@/components/ui/Stat'
import { VerdictCounts } from '@/components/ui/VerdictBadge'
import { useNow } from '@/hooks/useNow'
import { formatInt, formatPercent, ratio, SECONDS_PER_HOUR } from '@/lib/format'
import { countColumns, KIND_WORDS } from '@/lib/kinds'
import { Lockout } from './Lockout'

const BUDGET_CEILING = 100

function returnedDetail(kind: Kind, miner: MinerDetail): string {
  const { missing } = miner.kinds[kind].window
  if (kind === 'embed') return `${formatInt(missing)} texts missing`
  return miner.coverage.assigned === undefined
    ? `${formatInt(missing)} URLs missing`
    : `${formatPercent(miner.coverage.coverage)} of URLs returned · ${formatInt(missing)} missing`
}

export function MinerStats({ miner, kind }: { miner: MinerDetail; kind: Kind }) {
  const now = useNow()
  const pool = miner.pools[kind]
  const { window, share, claims } = miner.kinds[kind]
  const { returned, credited } = countColumns(kind)
  const hours = miner.window_hours
  const since = now / 1000 - hours * SECONDS_PER_HOUR
  const takenBack = miner.take_backs
    .filter((takeBack) => takeBack.kind === kind && takeBack.at >= since)
    .reduce((sum, takeBack) => sum + takeBack.tasks, 0)
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
      <Stat
        label="Share"
        value={formatPercent(share)}
        detail={`of the ${kind} pool · ${hours} h`}
      />
      <Stat
        label={`Tasks · ${hours} h`}
        value={formatInt(window.tasks)}
        detail={<VerdictCounts pass={window.pass} fail={window.fail} void={window.void} />}
      />
      <Stat
        label={`${returned.label} · ${hours} h`}
        value={formatInt(window.returned)}
        detail={returnedDetail(kind, miner)}
      />
      <Stat
        label={`${credited.label} · ${hours} h`}
        value={formatInt(window.credited)}
        detail={
          kind === 'embed'
            ? `${formatInt(pool.verified)} all time`
            : `${formatPercent(ratio(window.credited, window.returned))} of returned · ${formatInt(pool.verified)} all time`
        }
      />
      <Stat
        label={`Claimed · ${hours} h`}
        value={formatInt(claims.issued)}
        detail={`${formatInt(claims.completed)} uploaded · ${formatInt(claims.lapsed)} ran out or given back`}
      />
      <Stat
        label="Budget"
        value={`${formatInt(pool.budget)} of ${BUDGET_CEILING}`}
        detail={`tasks at once · ${formatInt(pool.in_flight)} ${KIND_WORDS[kind].working}, ${formatInt(pool.waiting)} waiting for a verdict`}
      />
      <Stat
        label="Lockout"
        value={<Lockout until={pool.locked_until} />}
        detail={`no new ${kind} tasks while locked out`}
      />
      <Stat
        label={`Taken back · ${hours} h`}
        value={formatInt(takenBack)}
        detail="passes taken back after a later check failed"
      />
    </div>
  )
}
