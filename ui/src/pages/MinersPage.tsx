import { useMiners } from '@/api/queries/miners'
import { MinersTable } from '@/components/miners/MinersTable'
import { Card } from '@/components/ui/Card'
import { KindSwitch } from '@/components/ui/KindSwitch'
import { PageHeader } from '@/components/ui/PageHeader'
import { QueryState } from '@/components/ui/QueryState'
import { EmptyState } from '@/components/ui/States'
import { useKindParam } from '@/hooks/useKindParam'

export function MinersPage() {
  const [kind, setKind] = useKindParam()
  const query = useMiners(kind)
  return (
    <>
      <PageHeader
        title="Miners"
        description="Share, budget and results per miner for one kind of task. Each kind has its own pool, budget and lockout. Counts cover the last 24 hours unless a column says otherwise. Select a column heading to sort."
      />
      <Card>
        <div className="border-line border-b px-4 py-3">
          <KindSwitch kind={kind} onChange={setKind} noun="miners" />
        </div>
        <QueryState query={query}>
          {({ miners }) =>
            miners.length === 0 ? (
              <EmptyState title="No miners to show yet.">
                Miners appear here once they have claimed a {kind} task.
              </EmptyState>
            ) : (
              <MinersTable key={kind} miners={miners} kind={kind} />
            )
          }
        </QueryState>
      </Card>
    </>
  )
}
