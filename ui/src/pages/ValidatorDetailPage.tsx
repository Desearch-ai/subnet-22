import { useParams } from 'react-router'
import { useValidator } from '@/api/queries/validators'
import { Card } from '@/components/ui/Card'
import { NeuronKeys } from '@/components/ui/NeuronKeys'
import { PageHeader } from '@/components/ui/PageHeader'
import { QueryState } from '@/components/ui/QueryState'
import { EmptyState } from '@/components/ui/States'
import { EXCLUDED_MEANING } from '@/lib/labels'
import { ValidatorStats } from '@/components/validators/ValidatorStats'
import { VoteList } from '@/components/validators/VoteList'

export function ValidatorDetailPage() {
  const { hotkey = '' } = useParams()
  const query = useValidator(hotkey)
  return (
    <div className="space-y-4">
      <PageHeader
        title="Validator"
        description={
          <NeuronKeys
            hotkey={hotkey}
            label="validator hotkey"
            uid={query.data?.uid ?? null}
            coldkey={query.data?.coldkey ?? null}
          />
        }
      />
      <QueryState query={query}>
        {(validator) =>
          validator.known ? (
            <>
              {validator.excluded ? (
                <p
                  role="status"
                  className="border-fail/40 bg-fail-wash text-ink rounded-lg border px-4 py-2.5 text-sm"
                >
                  {EXCLUDED_MEANING}.
                </p>
              ) : null}
              <ValidatorStats validator={validator} />
              <VoteList validator={hotkey} />
            </>
          ) : (
            <Card>
              <EmptyState title="The task API has never seen this hotkey as a validator.">
                Check the hotkey. A validator appears here once it has asked for work or voted.
              </EmptyState>
            </Card>
          )
        }
      </QueryState>
    </div>
  )
}
