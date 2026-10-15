import type { Kind, MinerDetail } from '@/api/types'
import { KindBadge } from '@/components/ui/KindBadge'
import { Stat } from '@/components/ui/Stat'
import { VerdictCounts } from '@/components/ui/VerdictBadge'
import { formatInt, formatPercent, ratio } from '@/lib/format'
import { countColumns, KIND_LABEL, KIND_SUMMARY, KINDS, KIND_WORDS } from '@/lib/kinds'
import { Lockout } from './Lockout'

function returnedDetail(kind: Kind, miner: MinerDetail): string {
  const { missing } = miner.kinds[kind].window
  if (kind === 'embed') return `${formatInt(missing)} texts missing`
  return miner.coverage.assigned === undefined
    ? `${formatInt(missing)} URLs missing`
    : `${formatPercent(miner.coverage.coverage)} coverage · ${formatInt(missing)} URLs missing`
}

function countedDetail(kind: Kind, miner: MinerDetail): string {
  const { window } = miner.kinds[kind]
  const allTime = `${formatInt(miner.pools[kind].verified)} all time`
  return kind === 'embed'
    ? allTime
    : `${formatPercent(ratio(window.credited, window.returned))} of rows returned · ${allTime}`
}

function KindStats({ kind, miner }: { kind: Kind; miner: MinerDetail }) {
  const pool = miner.pools[kind]
  const { window, share } = miner.kinds[kind]
  const { returned, credited } = countColumns(kind)
  const hours = miner.window_hours
  return (
    <section aria-label={`${KIND_LABEL[kind]} tasks`} className="min-w-0 space-y-2">
      <h2 className="text-ink-muted flex flex-wrap items-center gap-2 text-xs">
        <KindBadge kind={kind} />
        {KIND_SUMMARY[kind]}
      </h2>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        <Stat
          label="Share"
          value={formatPercent(share)}
          detail={`of the ${kind} pool · ${hours} h`}
        />
        <Stat
          label="Budget"
          value={formatInt(pool.budget)}
          detail={`tasks at once · ${formatInt(pool.in_flight)} ${KIND_WORDS[kind].working}, ${formatInt(pool.waiting)} waiting for a verdict`}
        />
        <Stat
          label="Lockout"
          value={<Lockout until={pool.locked_until} />}
          detail={`no new ${kind} tasks while locked out`}
        />
        <Stat
          label={`Tasks · ${hours} h`}
          value={formatInt(window.tasks)}
          detail={<VerdictCounts pass={window.pass} fail={window.fail} void={window.void} />}
        />
        <Stat
          label={returned.label}
          value={formatInt(window.returned)}
          detail={returnedDetail(kind, miner)}
        />
        <Stat
          label={credited.label}
          value={formatInt(window.credited)}
          detail={countedDetail(kind, miner)}
        />
      </div>
    </section>
  )
}

export function MinerStats({ miner }: { miner: MinerDetail }) {
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      {KINDS.map((kind) => (
        <KindStats key={kind} kind={kind} miner={miner} />
      ))}
    </div>
  )
}
