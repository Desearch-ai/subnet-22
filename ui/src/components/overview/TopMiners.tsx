import { Link } from 'react-router'
import { useMiners } from '@/api/queries/miners'
import type { MinerSummary } from '@/api/types'
import { Card, CardHeader } from '@/components/ui/Card'
import { Identifier } from '@/components/ui/Identifier'
import { QueryState } from '@/components/ui/QueryState'
import { EmptyState } from '@/components/ui/States'
import { formatInt, formatPercent } from '@/lib/format'
import { MINERS_PATH, minerPath } from '@/lib/paths'

const TOP_MINERS = 8

function ShareRow({ miner }: { miner: MinerSummary }) {
  return (
    <li className="px-4 py-2">
      <div className="flex items-center justify-between gap-3">
        <Identifier
          value={miner.hotkey}
          label="miner hotkey"
          uid={miner.uid}
          to={minerPath(miner.hotkey)}
        />
        <span className="text-ink font-mono text-xs tabular-nums">
          {formatPercent(miner.share)}
        </span>
      </div>
      <div className="bg-raised mt-1.5 h-1 overflow-hidden rounded-sm">
        <div className="bg-series-counted h-full" style={{ width: formatPercent(miner.share) }} />
      </div>
      <div className="text-ink-faint mt-1 text-xs">{formatInt(miner.credited)} rows counted</div>
    </li>
  )
}

export function TopMiners() {
  const query = useMiners()
  return (
    <Card>
      <CardHeader
        title="Top miners by share"
        description="Each miner's part of the crawl pool"
        actions={
          <Link to={MINERS_PATH} className="text-accent text-xs hover:underline">
            All miners
          </Link>
        }
      />
      <QueryState query={query}>
        {({ miners }) =>
          miners.length === 0 ? (
            <EmptyState title="No miner has a finalized task yet." />
          ) : (
            <ul className="divide-line divide-y">
              {miners.slice(0, TOP_MINERS).map((miner) => (
                <ShareRow key={miner.hotkey} miner={miner} />
              ))}
            </ul>
          )
        }
      </QueryState>
    </Card>
  )
}
