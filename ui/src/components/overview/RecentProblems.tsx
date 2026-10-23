import { Link } from 'react-router'
import { useTasks } from '@/api/queries/tasks'
import type { Kind } from '@/api/types'
import { TasksTable } from '@/components/tasks/TasksTable'
import { Card, CardHeader } from '@/components/ui/Card'
import { EmptyState, ErrorState, LoadingState } from '@/components/ui/States'
import { tasksPath } from '@/lib/paths'

const SHOWN = 10

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
        title="Latest fails and voids"
        description={`The newest ${kind} uploads that did not pass, with the reason`}
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
        <TasksTable tasks={tasks} kind={kind} />
      )}
    </Card>
  )
}
