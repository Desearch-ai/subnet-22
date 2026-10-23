import type { TakeBack } from '@/api/types'
import { KindBadge } from '@/components/ui/KindBadge'
import { EmptyState } from '@/components/ui/States'
import { Table, Td, Th, Tr } from '@/components/ui/Table'
import { TimeAgo } from '@/components/ui/TimeAgo'
import { formatInt } from '@/lib/format'
import { KIND_WORDS } from '@/lib/kinds'
import { TAKE_BACK_MEANING } from '@/lib/labels'

export function TakeBacks({ takeBacks }: { takeBacks: readonly TakeBack[] }) {
  if (takeBacks.length === 0) {
    return <EmptyState title="Nothing this miner passed was taken back in the last 8 days." />
  }
  return (
    <Table>
      <thead>
        <tr>
          <Th>When</Th>
          <Th>Kind</Th>
          <Th>Why</Th>
          <Th align="right" hint="Uploads that had passed and no longer count">
            Tasks
          </Th>
          <Th align="right" hint="What those uploads had counted toward the share">
            Uncounted
          </Th>
        </tr>
      </thead>
      <tbody>
        {takeBacks.map((takeBack) => (
          <Tr key={`${takeBack.at}:${takeBack.kind}`}>
            <Td className="text-ink-muted text-xs">
              <TimeAgo at={takeBack.at} />
            </Td>
            <Td>
              <KindBadge kind={takeBack.kind} />
            </Td>
            <Td className="min-w-64 whitespace-normal">
              <div className="font-mono text-xs">{takeBack.reason}</div>
              <div className="text-ink-muted text-xs">{TAKE_BACK_MEANING[takeBack.reason]}</div>
            </Td>
            <Td numeric>{formatInt(takeBack.tasks)}</Td>
            <Td numeric>
              {formatInt(takeBack.credited)}
              <span className="text-ink-faint"> {KIND_WORDS[takeBack.kind].credited}</span>
            </Td>
          </Tr>
        ))}
      </tbody>
    </Table>
  )
}
