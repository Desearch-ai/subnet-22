import type { TaskScore } from '@/api/types'
import { Identifier } from '@/components/ui/Identifier'
import { Table, Td, Th, Tr } from '@/components/ui/Table'
import { TimeAgo } from '@/components/ui/TimeAgo'
import { StatusBadge } from '@/components/ui/VerdictBadge'
import { formatCrawlTime, formatInt } from '@/lib/format'
import { explainReason } from '@/lib/labels'
import { minerPath, taskPath, validatorPath } from '@/lib/paths'

interface TasksTableProps {
  tasks: readonly TaskScore[]
  showMiner?: boolean
}

export function TasksTable({ tasks, showMiner = true }: TasksTableProps) {
  return (
    <Table>
      <thead>
        <tr>
          <Th>Task</Th>
          <Th>Result</Th>
          <Th>Reason</Th>
          {showMiner ? <Th>Miner</Th> : null}
          <Th hint="The validator whose report became the final result">Validator</Th>
          <Th align="right" hint="Rows the miner uploaded">
            Returned
          </Th>
          <Th align="right" hint="Assigned URLs with no row in the upload">
            Missing
          </Th>
          <Th align="right" hint="Pages the validator fetched again: matched / checked">
            Checked
          </Th>
          <Th align="right" hint="Rows that count toward the miner's share">
            Counted
          </Th>
          <Th align="right" hint="From the miner claiming the task to its upload being accepted">
            Crawl time
          </Th>
          <Th align="right">Finalized</Th>
        </tr>
      </thead>
      <tbody>
        {tasks.map((task) => (
          <Tr key={`${task.task_id}:${task.scored_at}`}>
            <Td>
              <Identifier value={task.task_id} label="task id" to={taskPath(task.task_id)} />
            </Td>
            <Td>
              <StatusBadge status={task.verdict} />
            </Td>
            <Td
              className="text-ink-muted font-mono text-xs"
              title={explainReason(task.reason) ?? undefined}
            >
              {task.reason}
            </Td>
            {showMiner ? (
              <Td>
                <Identifier
                  value={task.miner}
                  label="miner hotkey"
                  uid={task.miner_uid}
                  to={minerPath(task.miner)}
                />
              </Td>
            ) : null}
            <Td>
              <Identifier
                value={task.validator}
                label="validator hotkey"
                uid={task.validator_uid}
                to={validatorPath(task.validator)}
              />
            </Td>
            <Td numeric>{formatInt(task.returned)}</Td>
            <Td numeric className={task.missing > 0 ? undefined : 'text-ink-faint'}>
              {formatInt(task.missing)}
            </Td>
            <Td numeric>
              {task.matched}
              <span className="text-ink-faint"> / {task.sampled}</span>
            </Td>
            <Td numeric>{formatInt(task.credited)}</Td>
            <Td numeric>{formatCrawlTime(task)}</Td>
            <Td align="right" className="text-ink-muted text-xs">
              <TimeAgo at={task.scored_at} />
            </Td>
          </Tr>
        ))}
      </tbody>
    </Table>
  )
}
