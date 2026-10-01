import { useState } from 'react'
import type { MinerSummary } from '@/api/types'
import { Identifier } from '@/components/ui/Identifier'
import { SortTh } from '@/components/ui/SortTh'
import { Table, Td, Tr } from '@/components/ui/Table'
import { TimeAgo } from '@/components/ui/TimeAgo'
import { VerdictCounts } from '@/components/ui/VerdictBadge'
import { formatInt, formatPercent } from '@/lib/format'
import { minerPath } from '@/lib/paths'
import { nextSort, sortRows, type SortState, type SortValue } from '@/lib/sort'
import { Lockout } from './Lockout'

type MinerSortKey =
  | 'hotkey'
  | 'share'
  | 'budget'
  | 'in_flight'
  | 'waiting'
  | 'coverage'
  | 'pass'
  | 'returned'
  | 'credited'
  | 'verified'
  | 'locked_until'
  | 'last_scored_at'

const DEFAULT_SORT: SortState<MinerSortKey> = { key: 'share', direction: 'desc' }

function sortValue(miner: MinerSummary, key: MinerSortKey): SortValue {
  return miner[key]
}

export function MinersTable({ miners }: { miners: readonly MinerSummary[] }) {
  const [sort, setSort] = useState(DEFAULT_SORT)
  const onSort = (key: MinerSortKey) => {
    setSort(nextSort(sort, key))
  }
  const column = { sort, onSort }

  return (
    <Table>
      <thead>
        <tr>
          <SortTh {...column} sortKey="hotkey" label="Miner" align="left" />
          <SortTh
            {...column}
            sortKey="share"
            label="Share"
            hint="The miner's part of the crawl pool"
          />
          <SortTh
            {...column}
            sortKey="budget"
            label="Budget"
            hint="How many tasks the miner may hold at once"
          />
          <SortTh
            {...column}
            sortKey="in_flight"
            label="Crawling"
            hint="Tasks the miner holds and has not uploaded yet"
          />
          <SortTh
            {...column}
            sortKey="waiting"
            label="Waiting"
            hint="Uploads waiting for validators' verdict"
          />
          <SortTh
            {...column}
            sortKey="coverage"
            label="Coverage"
            hint="Rows returned out of URLs assigned"
          />
          <SortTh {...column} sortKey="pass" label="Pass / fail / void" align="left" />
          <SortTh {...column} sortKey="returned" label="Returned" hint="Rows the miner uploaded" />
          <SortTh
            {...column}
            sortKey="credited"
            label="Counted"
            hint="Rows that count toward the miner's share"
          />
          <SortTh
            {...column}
            sortKey="verified"
            label="Counted all time"
            hint="Rows counted since the start"
          />
          <SortTh
            {...column}
            sortKey="locked_until"
            label="Lockout"
            align="left"
            hint="A locked-out miner gets no new tasks until the lockout ends"
          />
          <SortTh {...column} sortKey="last_scored_at" label="Last finalized" />
        </tr>
      </thead>
      <tbody>
        {sortRows(miners, sort, sortValue).map((miner) => (
          <Tr key={miner.hotkey}>
            <Td>
              <Identifier
                value={miner.hotkey}
                label="miner hotkey"
                uid={miner.uid}
                to={minerPath(miner.hotkey)}
              />
            </Td>
            <Td numeric>{formatPercent(miner.share)}</Td>
            <Td numeric>{miner.budget}</Td>
            <Td numeric>{miner.in_flight}</Td>
            <Td numeric>{miner.waiting}</Td>
            <Td numeric>{formatPercent(miner.coverage)}</Td>
            <Td>
              <VerdictCounts pass={miner.pass} fail={miner.fail} void={miner.void} />
            </Td>
            <Td numeric>{formatInt(miner.returned)}</Td>
            <Td numeric>{formatInt(miner.credited)}</Td>
            <Td numeric>{formatInt(miner.verified)}</Td>
            <Td>
              <Lockout until={miner.locked_until} />
            </Td>
            <Td align="right" className="text-ink-muted text-xs">
              <TimeAgo at={miner.last_scored_at} />
            </Td>
          </Tr>
        ))}
      </tbody>
    </Table>
  )
}
