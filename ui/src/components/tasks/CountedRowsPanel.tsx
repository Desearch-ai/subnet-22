import type { TaskScore } from '@/api/types'
import { Card, CardHeader } from '@/components/ui/Card'
import { formatInt, formatPercent, ratio } from '@/lib/format'

interface StepProps {
  value: string
  label: string
  detail?: string
}

function Step({ value, label, detail }: StepProps) {
  return (
    <li className="border-line min-w-0 flex-1 basis-32 rounded-md border px-3 py-2">
      <div className="text-ink font-mono text-lg font-medium tabular-nums">{value}</div>
      <div className="text-ink-muted text-xs">{label}</div>
      {detail ? <div className="text-ink-faint text-xs">{detail}</div> : null}
    </li>
  )
}

export function CountedRowsPanel({ score }: { score: TaskScore }) {
  const assigned = score.returned + score.missing
  const checkedErrors = score.errors_confirmed + score.errors_unconfirmed
  const countedShare = ratio(score.credited, score.returned)
  return (
    <Card>
      <CardHeader
        title="From checked pages to rows counted"
        description="Validators fetch a sample of the pages again. What holds up in the sample decides how many of the returned rows count toward the miner's share."
      />
      <div className="p-4">
        <ol className="flex flex-wrap gap-2">
          <Step value={formatInt(assigned)} label="URLs assigned" />
          <Step
            value={formatInt(score.returned)}
            label="rows returned"
            detail={`${formatPercent(ratio(score.returned, assigned))} of assigned`}
          />
          <Step value={formatInt(score.sampled)} label="pages checked" />
          <Step
            value={formatInt(score.matched)}
            label="checked pages matched"
            detail="sets the share of content rows counted"
          />
          <Step
            value={`${formatInt(score.errors_confirmed)} / ${formatInt(checkedErrors)}`}
            label="checked errors confirmed"
            detail="sets the share of error rows counted"
          />
          <Step
            value={formatInt(score.credited)}
            label="rows counted"
            detail={`${formatPercent(countedShare)} of returned`}
          />
        </ol>
        <p className="text-ink-muted mt-3 text-sm">
          {score.verdict === 'pass'
            ? 'Rows with page content count at the share of checked pages that matched. Rows that reported an error count at the share of checked errors the validator confirmed. A mismatch in the sample therefore lowers the counted rows for the whole upload, not just for that page.'
            : 'No rows count for an upload that failed or ended void.'}
        </p>
      </div>
    </Card>
  )
}
