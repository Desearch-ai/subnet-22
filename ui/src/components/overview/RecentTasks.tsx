import { Link } from 'react-router'
import { useTasks } from '@/api/queries/tasks'
import { TasksTable } from '@/components/tasks/TasksTable'
import { Card, CardHeader } from '@/components/ui/Card'
import { EmptyState, ErrorState, LoadingState } from '@/components/ui/States'
import { TASKS_PATH } from '@/lib/paths'

const RECENT_TASKS = 10
const NO_FILTERS = {}

export function RecentTasks() {
  const query = useTasks(NO_FILTERS, RECENT_TASKS)
  const tasks = query.data?.pages[0]?.tasks
  return (
    <Card>
      <CardHeader
        title="Recent tasks"
        description="The latest finalized uploads"
        actions={
          <Link to={TASKS_PATH} className="text-accent text-xs hover:underline">
            All tasks
          </Link>
        }
      />
      {tasks === undefined ? (
        query.isError ? (
          <ErrorState error={query.error} />
        ) : (
          <LoadingState />
        )
      ) : tasks.length === 0 ? (
        <EmptyState title="No task has been finalized yet.">
          Tasks appear here once validators have voted on an upload.
        </EmptyState>
      ) : (
        <TasksTable tasks={tasks} />
      )}
    </Card>
  )
}
