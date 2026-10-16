import { useParams } from 'react-router'
import { useMiner } from '@/api/queries/miners'
import { SeriesPanel } from '@/components/charts/SeriesPanel'
import { BudgetHistory } from '@/components/miners/BudgetHistory'
import { MinerNotice } from '@/components/miners/MinerNotice'
import { MinerStats } from '@/components/miners/MinerStats'
import { TaskList } from '@/components/tasks/TaskList'
import { Card, CardHeader } from '@/components/ui/Card'
import { KindSwitch } from '@/components/ui/KindSwitch'
import { NeuronKeys } from '@/components/ui/NeuronKeys'
import { PageHeader } from '@/components/ui/PageHeader'
import { QueryState } from '@/components/ui/QueryState'
import { EmptyState } from '@/components/ui/States'
import { useKindParam } from '@/hooks/useKindParam'
import { KINDS } from '@/lib/kinds'

export function MinerDetailPage() {
  const { hotkey = '' } = useParams()
  const query = useMiner(hotkey)
  const [kind, setKind] = useKindParam()
  return (
    <div className="space-y-4">
      <PageHeader
        title="Miner"
        description={
          <NeuronKeys
            hotkey={hotkey}
            label="miner hotkey"
            uid={query.data?.uid ?? null}
            coldkey={query.data?.coldkey ?? null}
          />
        }
      />
      <QueryState query={query}>
        {(miner) =>
          miner.known ? (
            <>
              {KINDS.map((each) => (
                <MinerNotice key={each} kind={each} pool={miner.pools[each]} />
              ))}
              <MinerStats miner={miner} />
              <SeriesPanel
                kind={kind}
                miner={hotkey}
                actions={<KindSwitch kind={kind} onChange={setKind} />}
              />
              <Card>
                <CardHeader
                  title="Budget history"
                  description="How many tasks of each kind the miner may hold at once, and what changed it"
                />
                <BudgetHistory transitions={miner.transitions} />
              </Card>
              <Card>
                <CardHeader
                  title="Its tasks"
                  description="Finalized uploads of both kinds from this miner, newest first"
                />
                <TaskList
                  filters={{ miner: hotkey }}
                  showMiner={false}
                  emptyTitle="This miner has no finalized task yet."
                />
              </Card>
            </>
          ) : (
            <Card>
              <EmptyState title="The task API has never seen this hotkey as a miner.">
                Check the hotkey. A miner appears here once it has asked for a task.
              </EmptyState>
            </Card>
          )
        }
      </QueryState>
    </div>
  )
}
