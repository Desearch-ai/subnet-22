import type { TaskDetail, TaskScore, TaskStatus, Verdict } from '@/api/types'
import { Badge, type Tone } from '@/components/ui/Badge'
import { Card } from '@/components/ui/Card'
import { Field, FieldList } from '@/components/ui/Field'
import { Identifier } from '@/components/ui/Identifier'
import { EMPTY_VALUE } from '@/lib/format'
import { STATUS_MEANING } from '@/lib/labels'
import { minerPath } from '@/lib/paths'

type OpenStatus = Exclude<TaskStatus, Verdict>

const EARLIER_UPLOAD: Record<Verdict, string> = {
  pass: 'an upload that passed',
  fail: 'a failed upload',
  void: 'an upload that ended void',
}

const FIRST_ATTEMPT_TITLE: Record<OpenStatus, string> = {
  queued: 'Queued',
  claimed: 'Claimed',
  open: 'Uploaded, waiting for votes',
  voting: 'Being checked',
}

const RETRY_TITLE: Record<OpenStatus, string> = {
  queued: 'Queued again',
  claimed: 'Claimed again',
  open: 'Uploaded again, waiting for votes',
  voting: 'Being checked again',
}

const FINAL_TITLE: Record<Verdict, string> = {
  pass: 'Passed',
  fail: 'Failed',
  void: 'Ended void',
}

const STATUS_TONE: Record<TaskStatus, Tone> = {
  queued: 'neutral',
  claimed: 'accent',
  open: 'accent',
  voting: 'accent',
  pass: 'pass',
  fail: 'fail',
  void: 'void',
}

function isFinal(status: TaskStatus): status is Verdict {
  return status === 'pass' || status === 'fail' || status === 'void'
}

function statusTitle(status: TaskStatus, earlier: TaskScore | null): string {
  if (isFinal(status)) return FINAL_TITLE[status]
  if (earlier === null) return FIRST_ATTEMPT_TITLE[status]
  return `${RETRY_TITLE[status]} after ${EARLIER_UPLOAD[earlier.verdict]}`
}

export function TaskHeader({ task }: { task: TaskDetail }) {
  const { status } = task
  const earlier = task.uploads[0] ?? null
  const open = !isFinal(status)
  return (
    <Card>
      <div className="border-line border-b px-4 py-3">
        <Badge tone={STATUS_TONE[status]}>{statusTitle(status, earlier)}</Badge>
        <p className="text-ink-muted mt-1.5 text-sm">
          {STATUS_MEANING[status]}
          {open && earlier
            ? ' The result shown below is from an earlier upload of this task.'
            : null}
        </p>
      </div>
      <FieldList>
        <Field label="Task id">
          <Identifier value={task.task_id} label="task id" full />
        </Field>
        {open && status !== 'queued' ? (
          <Field label="Miner" hint="The miner working on the task right now">
            {task.miner === null ? (
              <span className="text-ink-faint">not recorded</span>
            ) : (
              <Identifier
                value={task.miner}
                label="miner hotkey"
                uid={task.miner_uid}
                to={minerPath(task.miner)}
                full
              />
            )}
          </Field>
        ) : null}
        <Field label="Round">
          <span className="font-mono text-xs">{task.round_id ?? EMPTY_VALUE}</span>
        </Field>
        <Field label="Finalized uploads" hint="How many uploads of this task have been checked">
          <span className="font-mono text-xs">{task.uploads.length}</span>
        </Field>
      </FieldList>
    </Card>
  )
}
