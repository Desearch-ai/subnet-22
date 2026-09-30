import type { TaskScore } from '@/api/types'
import { Card, CardHeader } from '@/components/ui/Card'
import { Field, FieldList } from '@/components/ui/Field'
import { Identifier } from '@/components/ui/Identifier'
import { TimeAgo } from '@/components/ui/TimeAgo'
import { StatusBadge } from '@/components/ui/VerdictBadge'
import { EMPTY_VALUE, formatCrawlTime } from '@/lib/format'
import { explainReason, VERDICT_MEANING } from '@/lib/labels'
import { minerPath, validatorPath } from '@/lib/paths'

interface UploadResultProps {
  score: TaskScore
  isLatestOfMany: boolean
}

export function UploadResult({ score, isLatestOfMany }: UploadResultProps) {
  return (
    <Card>
      <CardHeader
        title={isLatestOfMany ? 'Latest finalized upload' : 'Finalized upload'}
        description="The votes and URLs below belong to this upload and this miner"
      />
      <div className="border-line border-b px-4 py-3">
        <div className="flex flex-wrap items-center gap-2">
          <StatusBadge status={score.verdict} />
          <span className="text-ink-muted font-mono text-xs">{score.reason}</span>
        </div>
        <p className="text-ink-muted mt-1.5 text-sm">
          {explainReason(score.reason) ?? VERDICT_MEANING[score.verdict]}
        </p>
      </div>
      <FieldList>
        <Field label="Miner">
          <Identifier
            value={score.miner}
            label="miner hotkey"
            uid={score.miner_uid}
            to={minerPath(score.miner)}
            full
          />
        </Field>
        <Field label="Validator" hint="The validator whose report became the final result">
          <Identifier
            value={score.validator}
            label="validator hotkey"
            uid={score.validator_uid}
            to={validatorPath(score.validator)}
            full
          />
        </Field>
        <Field label="Claimed" hint="When the miner took the task">
          {score.claimed_at === null ? EMPTY_VALUE : <TimeAgo at={score.claimed_at} />}
        </Field>
        <Field label="Uploaded" hint="When the miner's upload was accepted for checking">
          {score.completed_at === null ? EMPTY_VALUE : <TimeAgo at={score.completed_at} />}
        </Field>
        <Field
          label="Crawl time"
          hint="From the miner claiming the task to its upload being accepted"
        >
          <span className="font-mono text-xs tabular-nums">{formatCrawlTime(score)}</span>
        </Field>
        <Field label="Finalized">
          <TimeAgo at={score.scored_at} />
        </Field>
        <Field label="Kind">
          <span className="font-mono text-xs">{score.kind}</span>
        </Field>
      </FieldList>
    </Card>
  )
}
