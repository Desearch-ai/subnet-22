import { useSearchParams } from 'react-router'
import type { TaskFilters } from '@/api/queries/tasks'
import type { Verdict } from '@/api/types'
import {
  FILTER_MINER,
  FILTER_VALIDATOR,
  FILTER_VERDICT,
  TaskFilterBar,
} from '@/components/tasks/TaskFilterBar'
import { TaskList } from '@/components/tasks/TaskList'
import { Card } from '@/components/ui/Card'
import { PageHeader } from '@/components/ui/PageHeader'
import { VERDICTS } from '@/lib/labels'

const FILTER_KEYS = [FILTER_MINER, FILTER_VALIDATOR, FILTER_VERDICT] as const

function readVerdict(value: string | null): Verdict | undefined {
  return VERDICTS.find((verdict) => verdict === value)
}

function readFilters(params: URLSearchParams): TaskFilters {
  const miner = params.get(FILTER_MINER)
  const validator = params.get(FILTER_VALIDATOR)
  const verdict = readVerdict(params.get(FILTER_VERDICT))
  return {
    ...(miner ? { miner } : {}),
    ...(validator ? { validator } : {}),
    ...(verdict ? { verdict } : {}),
  }
}

export function TasksPage() {
  const [params, setParams] = useSearchParams()
  const filters = readFilters(params)

  const apply = (form: FormData) => {
    const next = new URLSearchParams()
    for (const key of FILTER_KEYS) {
      const value = form.get(key)
      if (typeof value === 'string' && value.trim() !== '') next.set(key, value.trim())
    }
    setParams(next)
  }

  return (
    <>
      <PageHeader
        title="Tasks"
        description="Finalized uploads, newest first. A task that failed or ended void is queued again for another miner, so it can appear once per upload."
      />
      <Card>
        <TaskFilterBar
          filters={filters}
          onApply={apply}
          onClear={() => {
            setParams({})
          }}
        />
        <TaskList
          filters={filters}
          emptyTitle="No finalized task matches."
          emptyDetail="Change or clear the filters, or wait for validators to vote on the next upload."
        />
      </Card>
    </>
  )
}
