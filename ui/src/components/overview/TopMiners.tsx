import { Link } from 'react-router'
import { useMiners } from '@/api/queries/miners'
import type { Kind, MinerSummary } from '@/api/types'
import { Card, CardHeader } from '@/components/ui/Card'
import { Identifier } from '@/components/ui/Identifier'
import { QueryState } from '@/components/ui/QueryState'
import { EmptyState } from '@/components/ui/States'
import { formatInt, formatPercent } from '@/lib/format'
import { KIND_WORDS } from '@/lib/kinds'
import { MINERS_PATH, minerPath, withKind } from '@/lib/paths'

const TOP_MINERS = 8

function ShareRow({ miner, kind }: { miner: MinerSummary; kind: Kind }) {
  return (
    <li className="px-4 py-2">
      <div className="flex items-center justify-between gap-3">
        <Identifier
          value={miner.hotkey}
          label="miner hotkey"
          uid={miner.uid}
          to={withKind(minerPath(miner.hotkey), kind)}
        />
        <span className="text-ink font-mono text-xs tabular-nums">
          {formatPercent(miner.share)}
        </span>
      </div>
      <div className="bg-raised mt-1.5 h-1 overflow-hidden rounded-sm">
        <div className="bg-series-1 h-full" style={{ width: formatPercent(miner.share) }} />
      </div>
      <div className="text-ink-faint mt-1 text-xs">
        {formatInt(miner.credited)} {KIND_WORDS[kind].credited} counted
      </div>
    </li>
  )
}

export function TopMiners({ kind }: { kind: Kind }) {
  const query = useMiners(kind)
  return (
    <Card>
      <CardHeader
        title="Top miners by share"
        description={`Each miner's part of the ${kind} pool, last 24 hours`}
        actions={
          <Link to={withKind(MINERS_PATH, kind)} className="text-accent text-xs hover:underline">
            All miners
          </Link>
        }
      />
      <QueryState query={query}>
        {({ miners }) =>
          miners.length === 0 ? (
            <EmptyState title={`No miner has a finalized ${kind} task yet.`} />
          ) : (
            <ul className="divide-line divide-y">
              {miners.slice(0, TOP_MINERS).map((miner) => (
                <ShareRow key={miner.hotkey} miner={miner} kind={kind} />
              ))}
            </ul>
          )
        }
      </QueryState>
    </Card>
  )
}
