import { Link } from 'react-router'
import { useTasks } from '@/api/queries/tasks'
import type { Kind, TaskScore } from '@/api/types'
import { Card, CardHeader } from '@/components/ui/Card'
import { Neuron } from '@/components/ui/Neuron'
import { EmptyState, ErrorState, LoadingState } from '@/components/ui/States'
import { Table, Td, Th, Tr } from '@/components/ui/Table'
import { TimeAgo } from '@/components/ui/TimeAgo'
import { StatusBadge } from '@/components/ui/VerdictBadge'
import { explainReason } from '@/lib/labels'
import { minerPath, taskPath, tasksPath } from '@/lib/paths'

const SHOWN = 10

function ProblemsTable({ tasks }: { tasks: readonly TaskScore[] }) {
  return (
    <Table>
      <thead>
        <tr>
          <Th>When</Th>
          <Th>Result</Th>
          <Th>Why</Th>
          <Th>Miner</Th>
        </tr>
      </thead>
      <tbody>
        {tasks.map((task) => (
          <Tr key={`${task.task_id}:${task.scored_at}`}>
            <Td className="text-xs">
              <Link to={taskPath(task.task_id)} className="text-accent hover:underline">
                <TimeAgo at={task.scored_at} />
              </Link>
            </Td>
            <Td>
              <StatusBadge status={task.verdict} />
            </Td>
            <Td className="min-w-64 whitespace-normal" title={task.reason}>
              {explainReason(task.reason, task.kind) ?? task.reason}
            </Td>
            <Td>
              <Neuron
                hotkey={task.miner}
                label="miner hotkey"
                uid={task.miner_uid}
                coldkey={task.miner_coldkey}
                to={minerPath(task.miner)}
              />
            </Td>
          </Tr>
        ))}
      </tbody>
    </Table>
  )
}

export function RecentProblems({ kind }: { kind: Kind }) {
  const fails = useTasks({ verdict: 'fail', kind }, SHOWN)
  const voids = useTasks({ verdict: 'void', kind }, SHOWN)
  const failed = fails.isError ? fails : voids.isError ? voids : undefined
  const [failTasks, voidTasks] = [fails, voids].map((query) => query.data?.pages[0]?.tasks)
  const tasks =
    failTasks && voidTasks
      ? [...failTasks, ...voidTasks].sort((a, b) => b.scored_at - a.scored_at).slice(0, SHOWN)
      : undefined
  return (
    <Card>
      <CardHeader
        title="Latest uploads that did not pass"
        description="Newest first, with the reason"
        actions={
          <>
            <Link to={tasksPath('fail', kind)} className="text-accent text-xs hover:underline">
              All fails
            </Link>
            <Link to={tasksPath('void', kind)} className="text-accent text-xs hover:underline">
              All voids
            </Link>
          </>
        }
      />
      {tasks === undefined ? (
        failed ? (
          <ErrorState error={failed.error} />
        ) : (
          <LoadingState />
        )
      ) : tasks.length === 0 ? (
        <EmptyState title={`No ${kind} upload has failed or been voided.`} />
      ) : (
        <ProblemsTable tasks={tasks} />
      )}
    </Card>
  )
}
