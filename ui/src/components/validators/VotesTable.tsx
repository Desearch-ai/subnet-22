import type { Vote } from '@/api/types'
import { Badge } from '@/components/ui/Badge'
import { Identifier } from '@/components/ui/Identifier'
import { KindBadge } from '@/components/ui/KindBadge'
import { Table, Td, Th, Tr } from '@/components/ui/Table'
import { TimeAgo } from '@/components/ui/TimeAgo'
import { StatusBadge } from '@/components/ui/VerdictBadge'
import { formatInt } from '@/lib/format'
import { explainReason } from '@/lib/labels'
import { minerPath, taskPath } from '@/lib/paths'

function Agreement({ vote }: { vote: Vote }) {
  if (vote.agreed === null) return <span className="text-ink-faint text-xs">not compared</span>
  return vote.agreed ? <Badge tone="pass">Agreed</Badge> : <Badge tone="fail">Disagreed</Badge>
}

export function VotesTable({ votes }: { votes: readonly Vote[] }) {
  return (
    <Table>
      <thead>
        <tr>
          <Th>Task</Th>
          <Th>Kind</Th>
          <Th>Miner</Th>
          <Th>Its vote</Th>
          <Th>Reason</Th>
          <Th>Final result</Th>
          <Th hint="Whether this vote matched the final result">Match</Th>
          <Th hint="The final result was taken from this vote">Decided</Th>
          <Th
            align="right"
            hint="Matched / checked by this validator. Crawl: pages it fetched again. Embed: texts it embedded again"
          >
            Checked
          </Th>
          <Th
            align="right"
            hint="What this validator would count / what finally counted. Crawl: rows. Embed: characters"
          >
            Counted
          </Th>
          <Th align="right">Voted</Th>
        </tr>
      </thead>
      <tbody>
        {votes.map((vote) => (
          <Tr key={`${vote.task_id}:${vote.voted_at}`}>
            <Td>
              <Identifier value={vote.task_id} label="task id" to={taskPath(vote.task_id)} />
            </Td>
            <Td>
              <KindBadge kind={vote.kind} />
            </Td>
            <Td>
              <Identifier
                value={vote.miner}
                label="miner hotkey"
                uid={vote.miner_uid}
                to={minerPath(vote.miner)}
              />
            </Td>
            <Td>
              <StatusBadge status={vote.verdict} />
            </Td>
            <Td
              className="text-ink-muted font-mono text-xs"
              title={explainReason(vote.reason, vote.kind) ?? undefined}
            >
              {vote.reason}
            </Td>
            <Td>
              <StatusBadge status={vote.final_verdict} />
            </Td>
            <Td>
              <Agreement vote={vote} />
            </Td>
            <Td>{vote.decided ? <Badge tone="accent">Decided</Badge> : null}</Td>
            <Td numeric>
              {vote.matched}
              <span className="text-ink-faint"> / {vote.sampled}</span>
            </Td>
            <Td numeric>
              {formatInt(vote.credited)}
              <span className="text-ink-faint"> / {formatInt(vote.final_credited)}</span>
            </Td>
            <Td align="right" className="text-ink-muted text-xs">
              <TimeAgo at={vote.voted_at} />
            </Td>
          </Tr>
        ))}
      </tbody>
    </Table>
  )
}
