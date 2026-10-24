import { Link } from 'react-router'
import { useMiners } from '@/api/queries/miners'
import type { Kind, MinerSummary } from '@/api/types'
import { Card, CardHeader } from '@/components/ui/Card'
import { Identifier } from '@/components/ui/Identifier'
import { QueryState } from '@/components/ui/QueryState'
import { EmptyState } from '@/components/ui/States'
import { formatPercent } from '@/lib/format'
import { MINERS_PATH, minerPath, withKind } from '@/lib/paths'

const TOP_MINERS = 8

/** The bar is drawn against the largest share, so close shares still look apart. */
function ShareRow({ miner, kind, largest }: { miner: MinerSummary; kind: Kind; largest: number }) {
  return (
    <li className="flex items-center gap-3 px-4 py-2.5">
      <Identifier
        value={miner.hotkey}
        label="miner hotkey"
        uid={miner.uid}
        to={withKind(minerPath(miner.hotkey), kind)}
        className="w-32 shrink-0"
      />
      <div className="bg-raised h-1.5 min-w-0 flex-1 overflow-hidden rounded-sm">
        <div
          className="bg-series-1 h-full rounded-sm"
          style={{ width: formatPercent(largest > 0 ? miner.share / largest : 0) }}
        />
      </div>
      <span className="text-ink w-12 shrink-0 text-right font-mono text-xs tabular-nums">
        {formatPercent(miner.share)}
      </span>
    </li>
  )
}

export function TopMiners({ kind }: { kind: Kind }) {
  const query = useMiners(kind)
  return (
    <Card>
      <CardHeader
        title="Top miners"
        description={`Share of the ${kind} pool, last 24 hours`}
        actions={
          <Link to={withKind(MINERS_PATH, kind)} className="text-accent text-xs hover:underline">
            All miners
          </Link>
        }
      />
      <QueryState query={query}>
        {({ miners }) => {
          const top = miners.slice(0, TOP_MINERS)
          return top.length === 0 ? (
            <EmptyState title={`No miner has a finalized ${kind} task yet.`} />
          ) : (
            <ul className="divide-line divide-y">
              {top.map((miner) => (
                <ShareRow
                  key={miner.hotkey}
                  miner={miner}
                  kind={kind}
                  largest={top[0]?.share ?? 0}
                />
              ))}
            </ul>
          )
        }}
      </QueryState>
    </Card>
  )
}
