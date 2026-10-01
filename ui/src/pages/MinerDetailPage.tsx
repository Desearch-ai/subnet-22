import { useParams } from 'react-router'
import { useMiner } from '@/api/queries/miners'
import { BudgetHistory } from '@/components/miners/BudgetHistory'
import { MinerStats } from '@/components/miners/MinerStats'
import { MinerNotice } from '@/components/miners/MinerNotice'
import { MinerRowsPanel } from '@/components/miners/MinerRowsPanel'
import { TaskList } from '@/components/tasks/TaskList'
import { Card, CardHeader } from '@/components/ui/Card'
import { NeuronKeys } from '@/components/ui/NeuronKeys'
import { PageHeader } from '@/components/ui/PageHeader'
import { QueryState } from '@/components/ui/QueryState'
import { EmptyState } from '@/components/ui/States'

export function MinerDetailPage() {
  const { hotkey = '' } = useParams()
  const query = useMiner(hotkey)
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
              <MinerNotice pool={miner.pools.crawl} />
              <MinerStats miner={miner} />
              <MinerRowsPanel miner={hotkey} />
              <Card>
                <CardHeader
                  title="Budget history"
                  description="How many tasks the miner may hold at once, and what changed it"
                />
                <BudgetHistory transitions={miner.transitions} />
              </Card>
              <Card>
                <CardHeader
                  title="Its tasks"
                  description="Finalized uploads from this miner, newest first"
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
