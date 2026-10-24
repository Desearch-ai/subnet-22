import { useState } from 'react'
import type { Kind, MinerSummary } from '@/api/types'
import { Neuron } from '@/components/ui/Neuron'
import { SortTh } from '@/components/ui/SortTh'
import { Table, Td, Tr } from '@/components/ui/Table'
import { TimeAgo } from '@/components/ui/TimeAgo'
import { VerdictCounts } from '@/components/ui/VerdictBadge'
import { formatInt, formatPercent } from '@/lib/format'
import { countColumns, KIND_LABEL, KIND_WORDS } from '@/lib/kinds'
import { minerPath, withKind } from '@/lib/paths'
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

const VERIFIED_HINT: Record<Kind, string> = {
  crawl: 'Pages counted since the start',
  embed: 'Characters counted since the start',
}

interface MinersTableProps {
  miners: readonly MinerSummary[]
  kind: Kind
}

export function MinersTable({ miners, kind }: MinersTableProps) {
  const [sort, setSort] = useState(DEFAULT_SORT)
  const onSort = (key: MinerSortKey) => {
    setSort(nextSort(sort, key))
  }
  const column = { sort, onSort }
  const { returned, credited } = countColumns(kind)
  const working = KIND_WORDS[kind].working
  const hasCoverage = kind === 'crawl'

  return (
    <Table>
      <thead>
        <tr>
          <SortTh {...column} sortKey="hotkey" label="Miner" align="left" />
          <SortTh
            {...column}
            sortKey="share"
            label="Share"
            hint={`The miner's part of the ${kind} pool, last 24 hours`}
          />
          <SortTh
            {...column}
            sortKey="budget"
            label="Budget"
            hint={`How many ${kind} tasks the miner may hold at once`}
          />
          <SortTh
            {...column}
            sortKey="in_flight"
            label={working}
            hint={`${KIND_LABEL[kind]} tasks the miner holds and has not uploaded yet`}
          />
          <SortTh
            {...column}
            sortKey="waiting"
            label="Waiting"
            hint="Uploads waiting for validators' verdict"
          />
          {hasCoverage ? (
            <SortTh
              {...column}
              sortKey="coverage"
              label="Coverage"
              hint="Pages returned out of URLs assigned"
            />
          ) : null}
          <SortTh {...column} sortKey="pass" label="Pass / fail / void" align="left" />
          <SortTh {...column} sortKey="returned" label={returned.label} hint={returned.hint} />
          <SortTh {...column} sortKey="credited" label={credited.label} hint={credited.hint} />
          <SortTh
            {...column}
            sortKey="verified"
            label="Counted all time"
            hint={VERIFIED_HINT[kind]}
          />
          <SortTh
            {...column}
            sortKey="locked_until"
            label="Lockout"
            align="left"
            hint={`A locked-out miner gets no new ${kind} tasks until the lockout ends`}
          />
          <SortTh {...column} sortKey="last_scored_at" label="Last finalized" />
        </tr>
      </thead>
      <tbody>
        {sortRows(miners, sort, sortValue).map((miner) => (
          <Tr key={miner.hotkey}>
            <Td>
              <Neuron
                hotkey={miner.hotkey}
                label="miner hotkey"
                uid={miner.uid}
                coldkey={miner.coldkey}
                to={withKind(minerPath(miner.hotkey), kind)}
              />
            </Td>
            <Td numeric>{formatPercent(miner.share)}</Td>
            <Td numeric>{miner.budget}</Td>
            <Td numeric>{miner.in_flight}</Td>
            <Td numeric>{miner.waiting}</Td>
            {hasCoverage ? <Td numeric>{formatPercent(miner.coverage)}</Td> : null}
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
