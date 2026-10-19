import type { TextOutcome, UrlOutcome } from '@/api/types'
import { Badge, type Tone } from '@/components/ui/Badge'
import { URL_OUTCOME_LABEL } from '@/lib/labels'

type Outcome = UrlOutcome | TextOutcome

const OUTCOME_TONE: Record<Outcome, Tone> = {
  matched: 'pass',
  mismatched: 'fail',
  malformed: 'fail',
  unverifiable: 'void',
  errors_confirmed: 'neutral',
  errors_unconfirmed: 'fail',
  not_fetched: 'neutral',
}

const OUTCOME_LABEL: Record<Outcome, string> = { ...URL_OUTCOME_LABEL, malformed: 'Malformed' }

export function OutcomeBadge({ outcome }: { outcome: Outcome }) {
  return <Badge tone={OUTCOME_TONE[outcome]}>{OUTCOME_LABEL[outcome]}</Badge>
}
