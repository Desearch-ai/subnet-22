import type { ValidatorSummary } from '@/api/types'
import { Identifier } from '@/components/ui/Identifier'
import { Table, Td, Th, Tr } from '@/components/ui/Table'
import { TimeAgo } from '@/components/ui/TimeAgo'
import { VerdictCounts } from '@/components/ui/VerdictBadge'
import { formatInt, formatPercent } from '@/lib/format'
import { validatorPath } from '@/lib/paths'
import { ValidatorFlags } from './ValidatorFlags'

export function ValidatorsTable({ validators }: { validators: readonly ValidatorSummary[] }) {
  return (
    <Table>
      <thead>
        <tr>
          <Th>Validator</Th>
          <Th>State</Th>
          <Th align="right">Last seen</Th>
          <Th align="right" hint="Votes cast in the window">
            Votes
          </Th>
          <Th>Pass / fail / void</Th>
          <Th
            align="right"
            hint="Votes that matched the final result, out of votes that could be compared"
          >
            Agreement
          </Th>
          <Th align="right" hint="Votes that did not match the final result, in the window">
            Disagreed
          </Th>
          <Th align="right" hint="Tasks whose final result was taken from this validator's report">
            Decided
          </Th>
          <Th align="right" hint="Uploads this validator has checked, all time">
            Checks all time
          </Th>
          <Th align="right" hint="Checks that did not match the final result, all time">
            Disagreements all time
          </Th>
          <Th align="right">Last vote</Th>
        </tr>
      </thead>
      <tbody>
        {validators.map((validator) => (
          <Tr key={validator.hotkey}>
            <Td>
              <Identifier
                value={validator.hotkey}
                label="validator hotkey"
                uid={validator.uid}
                to={validatorPath(validator.hotkey)}
              />
            </Td>
            <Td>
              <ValidatorFlags validator={validator} />
            </Td>
            <Td align="right" className="text-ink-muted text-xs">
              <TimeAgo at={validator.last_seen} />
            </Td>
            <Td numeric>{formatInt(validator.votes)}</Td>
            <Td>
              <VerdictCounts pass={validator.pass} fail={validator.fail} void={validator.void} />
            </Td>
            <Td numeric>{formatPercent(validator.agreement)}</Td>
            <Td numeric>{formatInt(validator.disagreed)}</Td>
            <Td numeric>{formatInt(validator.decided)}</Td>
            <Td numeric>{formatInt(validator.audits)}</Td>
            <Td numeric>{formatInt(validator.disagreements)}</Td>
            <Td align="right" className="text-ink-muted text-xs">
              <TimeAgo at={validator.last_vote_at} />
            </Td>
          </Tr>
        ))}
      </tbody>
    </Table>
  )
}
