import { useParams } from 'react-router'
import { ApiError } from '@/api/client'
import { useTask } from '@/api/queries/tasks'
import type { TaskDetail, TaskScore } from '@/api/types'
import { CountedRowsPanel } from '@/components/tasks/CountedRowsPanel'
import { TaskHeader } from '@/components/tasks/TaskHeader'
import { UploadResult } from '@/components/tasks/UploadResult'
import { UploadsTable } from '@/components/tasks/UploadsTable'
import { UrlTable } from '@/components/tasks/UrlTable'
import { VotesComparison } from '@/components/tasks/VotesComparison'
import { Card, CardHeader } from '@/components/ui/Card'
import { PageHeader } from '@/components/ui/PageHeader'
import { QueryState } from '@/components/ui/QueryState'
import { EmptyState } from '@/components/ui/States'
import { truncateMiddle } from '@/lib/format'

interface UploadSectionsProps {
  score: TaskScore
  task: TaskDetail
}

function UploadSections({ score, task }: UploadSectionsProps) {
  const by = `upload by ${truncateMiddle(score.miner)}`
  return (
    <>
      <UploadResult score={score} isLatestOfMany={task.uploads.length > 1} />
      <CountedRowsPanel score={score} />
      <Card>
        <CardHeader
          title="Validator votes"
          description={`Every validator checked the ${by} on its own and voted`}
        />
        <VotesComparison votes={task.votes} />
      </Card>
      <Card>
        <CardHeader
          title="URLs"
          description={`What the ${by} returned for each URL and what the validator found`}
        />
        {task.urls.length === 0 ? (
          <EmptyState title="No per-URL results to show.">
            They are kept for 7 days after a task is finalized and are not stored for a void task.
          </EmptyState>
        ) : (
          <UrlTable urls={task.urls} />
        )}
      </Card>
    </>
  )
}

export function TaskDetailPage() {
  const { taskId = '' } = useParams()
  const query = useTask(taskId)

  if (query.error instanceof ApiError && query.error.isNotFound) {
    return (
      <Card>
        <EmptyState title="No task has this id.">
          Check the id, or look the task up from a miner's page.
        </EmptyState>
      </Card>
    )
  }

  return (
    <div className="space-y-4">
      <PageHeader title="Task" />
      <QueryState query={query}>
        {(task) => (
          <>
            <TaskHeader task={task} />
            {task.uploads.length > 1 ? (
              <Card>
                <CardHeader
                  title="All uploads of this task"
                  description="A failed or void upload sends the task back to the queue for another miner"
                />
                <UploadsTable uploads={task.uploads} />
              </Card>
            ) : null}
            {task.score ? <UploadSections score={task.score} task={task} /> : null}
          </>
        )}
      </QueryState>
    </div>
  )
}
