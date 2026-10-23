import { Link } from 'react-router'
import { useValidators } from '@/api/queries/validators'
import { Card, CardHeader } from '@/components/ui/Card'
import { Identifier } from '@/components/ui/Identifier'
import { QueryState } from '@/components/ui/QueryState'
import { EmptyState } from '@/components/ui/States'
import { Table, Td, Th, Tr } from '@/components/ui/Table'
import { TimeAgo } from '@/components/ui/TimeAgo'
import { ValidatorFlags } from '@/components/validators/ValidatorFlags'
import { formatInt, formatPercent } from '@/lib/format'
import { VALIDATORS_PATH, validatorPath } from '@/lib/paths'

export function ValidatorsPanel() {
  const query = useValidators()
  return (
    <Card>
      <CardHeader
        title="Validators"
        description="Who is checking uploads now, and how often each agrees with the final result, last 24 hours"
        actions={
          <Link to={VALIDATORS_PATH} className="text-accent text-xs hover:underline">
            All validators
          </Link>
        }
      />
      <QueryState query={query}>
        {({ validators }) =>
          validators.length === 0 ? (
            <EmptyState title="No validator has asked for work or voted yet." />
          ) : (
            <Table>
              <thead>
                <tr>
                  <Th>Validator</Th>
                  <Th>State</Th>
                  <Th align="right">Votes</Th>
                  <Th
                    align="right"
                    hint="Votes that matched the final result, out of votes that could be compared"
                  >
                    Agrees
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
                    <Td numeric>{formatInt(validator.votes)}</Td>
                    <Td numeric>{formatPercent(validator.agreement)}</Td>
                    <Td align="right" className="text-ink-muted text-xs">
                      <TimeAgo at={validator.last_vote_at} />
                    </Td>
                  </Tr>
                ))}
              </tbody>
            </Table>
          )
        }
      </QueryState>
    </Card>
  )
}
