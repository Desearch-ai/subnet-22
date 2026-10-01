import type { TaskStatus, Verdict } from '@/api/types'
import { STATUS_LABEL, STATUS_MEANING } from '@/lib/labels'
import { Badge, Dot, type Tone } from './Badge'

const STATUS_TONE: Record<TaskStatus, Tone> = {
  queued: 'neutral',
  claimed: 'accent',
  open: 'accent',
  voting: 'accent',
  pass: 'pass',
  fail: 'fail',
  void: 'void',
}

export function StatusBadge({ status }: { status: TaskStatus }) {
  return (
    <Badge tone={STATUS_TONE[status]} title={STATUS_MEANING[status]}>
      {STATUS_LABEL[status]}
    </Badge>
  )
}

interface VerdictCountsProps {
  pass: number
  fail: number
  void: number
}

const COUNT_TONES: readonly Verdict[] = ['pass', 'fail', 'void']

export function VerdictCounts(counts: VerdictCountsProps) {
  return (
    <span className="inline-flex items-center gap-2.5 font-mono text-xs tabular-nums">
      {COUNT_TONES.map((verdict) => (
        <span
          key={verdict}
          title={STATUS_LABEL[verdict]}
          className="inline-flex items-center gap-1"
        >
          <Dot tone={verdict} />
          {counts[verdict]}
        </span>
      ))}
    </span>
  )
}
