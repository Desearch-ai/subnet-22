import { useMiners } from '@/api/queries/miners'
import { MinersTable } from '@/components/miners/MinersTable'
import { Card } from '@/components/ui/Card'
import { PageHeader } from '@/components/ui/PageHeader'
import { QueryState } from '@/components/ui/QueryState'
import { EmptyState } from '@/components/ui/States'

export function MinersPage() {
  const query = useMiners()
  return (
    <>
      <PageHeader
        title="Miners"
        description="Share, budget and results per miner. Counts cover the last 24 hours unless a column says otherwise. Select a column heading to sort."
      />
      <Card>
        <QueryState query={query}>
          {({ miners }) =>
            miners.length === 0 ? (
              <EmptyState title="No miners to show yet.">
                Miners appear here once they have claimed a task.
              </EmptyState>
            ) : (
              <MinersTable miners={miners} />
            )
          }
        </QueryState>
      </Card>
    </>
  )
}
