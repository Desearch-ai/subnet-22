import { Badge } from '@/components/ui/Badge'
import { EMPTY_VALUE } from '@/lib/format'

export const ELIGIBILITY_HINT =
  'Rows count only while the miner returns rows for at least 85% of the URLs it was assigned'

interface EligibilityProps {
  coverage: number | null | undefined
  eligible: boolean | undefined
}

export function Eligibility({ coverage, eligible }: EligibilityProps) {
  if (coverage === null || coverage === undefined || eligible === undefined) {
    return <span className="text-ink-faint">{EMPTY_VALUE}</span>
  }
  return eligible ? (
    <Badge tone="pass" title={ELIGIBILITY_HINT}>
      Eligible
    </Badge>
  ) : (
    <Badge tone="fail" title={ELIGIBILITY_HINT}>
      Not eligible
    </Badge>
  )
}
