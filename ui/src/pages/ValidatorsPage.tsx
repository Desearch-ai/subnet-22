import { useValidators } from '@/api/queries/validators'
import { PageHeader } from '@/components/ui/PageHeader'
import { Card } from '@/components/ui/Card'
import { QueryState } from '@/components/ui/QueryState'
import { EmptyState } from '@/components/ui/States'
import { ValidatorsTable } from '@/components/validators/ValidatorsTable'

export function ValidatorsPage() {
  const query = useValidators()
  return (
    <>
      <PageHeader
        title="Validators"
        description="Who is checking uploads, how often each one matches the final result, and who decided it. Counts cover the last 24 hours unless a column says otherwise."
      />
      <Card>
        <QueryState query={query}>
          {({ validators }) =>
            validators.length === 0 ? (
              <EmptyState title="No validators to show yet.">
                Validators appear here once they ask for work or vote.
              </EmptyState>
            ) : (
              <ValidatorsTable validators={validators} />
            )
          }
        </QueryState>
      </Card>
    </>
  )
}
