import type { UrlOutcome } from '@/api/types'
import { Badge, type Tone } from '@/components/ui/Badge'
import { URL_OUTCOME_LABEL } from '@/lib/labels'

const OUTCOME_TONE: Record<UrlOutcome, Tone> = {
  matched: 'pass',
  mismatched: 'fail',
  unverifiable: 'void',
  errors_confirmed: 'neutral',
  errors_unconfirmed: 'fail',
  not_fetched: 'neutral',
}

export function OutcomeBadge({ outcome }: { outcome: UrlOutcome }) {
  return <Badge tone={OUTCOME_TONE[outcome]}>{URL_OUTCOME_LABEL[outcome]}</Badge>
}
