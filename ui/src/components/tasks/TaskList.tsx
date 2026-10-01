import { useTasks, type TaskFilters } from '@/api/queries/tasks'
import { PagedList } from '@/components/ui/PagedList'
import { TasksTable } from './TasksTable'

interface TaskListProps {
  filters: TaskFilters
  showMiner?: boolean
  emptyTitle: string
  emptyDetail?: string
}

export function TaskList({ filters, showMiner = true, emptyTitle, emptyDetail }: TaskListProps) {
  const query = useTasks(filters)
  return (
    <PagedList
      query={query}
      itemsOf={(page) => page.tasks}
      emptyTitle={emptyTitle}
      {...(emptyDetail ? { emptyDetail } : {})}
    >
      {(tasks) => <TasksTable tasks={tasks} showMiner={showMiner} />}
    </PagedList>
  )
}
