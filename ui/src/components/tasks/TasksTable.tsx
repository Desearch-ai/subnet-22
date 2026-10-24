import type { Kind, TaskScore } from '@/api/types'
import { Identifier } from '@/components/ui/Identifier'
import { Neuron } from '@/components/ui/Neuron'
import { KindBadge } from '@/components/ui/KindBadge'
import { Table, Td, Th, Tr } from '@/components/ui/Table'
import { TimeAgo } from '@/components/ui/TimeAgo'
import { StatusBadge } from '@/components/ui/VerdictBadge'
import { formatInt, formatWorkTime } from '@/lib/format'
import { countColumns } from '@/lib/kinds'
import { explainReason } from '@/lib/labels'
import { minerPath, taskPath, validatorPath } from '@/lib/paths'

interface TasksTableProps {
  tasks: readonly TaskScore[]
  kind?: Kind | undefined
  showMiner?: boolean
}

export function TasksTable({ tasks, kind, showMiner = true }: TasksTableProps) {
  const columns = countColumns(kind)
  return (
    <Table>
      <thead>
        <tr>
          <Th>Task</Th>
          <Th>Kind</Th>
          <Th>Result</Th>
          <Th>Reason</Th>
          {showMiner ? <Th>Miner</Th> : null}
          <Th hint="The validator whose report became the final result">Validator</Th>
          <Th align="right" hint={columns.returned.hint}>
            {columns.returned.label}
          </Th>
          <Th align="right" hint={columns.missing.hint}>
            {columns.missing.label}
          </Th>
          <Th align="right" hint={columns.checked.hint}>
            {columns.checked.label}
          </Th>
          <Th align="right" hint={columns.credited.hint}>
            {columns.credited.label}
          </Th>
          <Th align="right" hint={columns.time.hint}>
            {columns.time.label}
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
              <KindBadge kind={task.kind} />
            </Td>
            <Td>
              <StatusBadge status={task.verdict} />
            </Td>
            <Td
              className="text-ink-muted font-mono text-xs"
              title={explainReason(task.reason, task.kind) ?? undefined}
            >
              {task.reason}
            </Td>
            {showMiner ? (
              <Td>
                <Neuron
                  hotkey={task.miner}
                  label="miner hotkey"
                  uid={task.miner_uid}
                  coldkey={task.miner_coldkey}
                  to={minerPath(task.miner)}
                />
              </Td>
            ) : null}
            <Td>
              <Neuron
                hotkey={task.validator}
                label="validator hotkey"
                uid={task.validator_uid}
                coldkey={task.validator_coldkey}
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
            <Td numeric>{formatWorkTime(task)}</Td>
            <Td align="right" className="text-ink-muted text-xs">
              <TimeAgo at={task.scored_at} />
            </Td>
          </Tr>
        ))}
      </tbody>
    </Table>
  )
}
