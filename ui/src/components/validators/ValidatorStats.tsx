import { WINDOW_HOURS } from '@/api/config'
import type { ValidatorSummary } from '@/api/types'
import { Stat, StatGrid } from '@/components/ui/Stat'
import { TimeAgo } from '@/components/ui/TimeAgo'
import { VerdictCounts } from '@/components/ui/VerdictBadge'
import { formatInt, formatPercent, ratio } from '@/lib/format'
import { ValidatorFlags } from './ValidatorFlags'

export function ValidatorStats({ validator }: { validator: ValidatorSummary }) {
  return (
    <StatGrid>
      <Stat
        label="State"
        value={<ValidatorFlags validator={validator} />}
        detail={
          <>
            last seen <TimeAgo at={validator.last_seen} />
          </>
        }
      />
      <Stat
        label={`Votes · ${WINDOW_HOURS} h`}
        value={formatInt(validator.votes)}
        detail={<VerdictCounts pass={validator.pass} fail={validator.fail} void={validator.void} />}
      />
      <Stat
        label="Agreement"
        value={formatPercent(validator.agreement)}
        detail={`${formatInt(validator.agreed)} agreed · ${formatInt(validator.disagreed)} disagreed`}
      />
      <Stat
        label="Decided"
        value={formatInt(validator.decided)}
        detail="tasks whose result came from its report"
      />
      <Stat
        label="Checks all time"
        value={formatInt(validator.audits)}
        detail={`${formatInt(validator.disagreements)} disagreements (${formatPercent(ratio(validator.disagreements, validator.audits))})`}
      />
      <Stat
        label="Last vote"
        value={
          <span className="text-base">
            <TimeAgo at={validator.last_vote_at} />
          </span>
        }
        detail="most recent vote cast"
      />
    </StatGrid>
  )
}
