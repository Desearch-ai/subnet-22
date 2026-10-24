import { useParams, useSearchParams } from 'react-router'
import { useMiner } from '@/api/queries/miners'
import type { Kind, Verdict } from '@/api/types'
import { WorkPanel } from '@/components/charts/WorkPanel'
import { BudgetCuts } from '@/components/miners/BudgetCuts'
import { MinerNotice } from '@/components/miners/MinerNotice'
import { MinerStats } from '@/components/miners/MinerStats'
import { TakeBacks } from '@/components/miners/TakeBacks'
import { TaskList } from '@/components/tasks/TaskList'
import { Card, CardHeader } from '@/components/ui/Card'
import { KindSwitch } from '@/components/ui/KindSwitch'
import { NeuronKeys } from '@/components/ui/NeuronKeys'
import { PageHeader } from '@/components/ui/PageHeader'
import { QueryState } from '@/components/ui/QueryState'
import { SegmentedControl } from '@/components/ui/SegmentedControl'
import { EmptyState } from '@/components/ui/States'
import { useKindParam } from '@/hooks/useKindParam'
import { KINDS } from '@/lib/kinds'
import { VERDICTS } from '@/lib/labels'

type Result = 'all' | Verdict

const RESULT_PARAM = 'result'
const RESULTS: readonly { value: Result; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'fail', label: 'Failed' },
  { value: 'void', label: 'Void' },
  { value: 'pass', label: 'Passed' },
]
const NONE_WITH: Record<Verdict, string> = { pass: 'passed', fail: 'failed', void: 'void' }

function useResultParam(): [Result, (result: Result) => void] {
  const [params, setParams] = useSearchParams()
  const result = VERDICTS.find((verdict) => verdict === params.get(RESULT_PARAM)) ?? 'all'
  const setResult = (next: Result) => {
    setParams(
      (current) => {
        const updated = new URLSearchParams(current)
        if (next === 'all') updated.delete(RESULT_PARAM)
        else updated.set(RESULT_PARAM, next)
        return updated
      },
      { replace: true },
    )
  }
  return [result, setResult]
}

function ItsTasks({ hotkey, kind }: { hotkey: string; kind: Kind }) {
  const [result, setResult] = useResultParam()
  return (
    <Card>
      <CardHeader
        title="Its tasks"
        description={`Its finalized ${kind} uploads, newest first`}
        actions={
          <SegmentedControl
            label="Show tasks"
            segments={RESULTS}
            value={result}
            onChange={setResult}
          />
        }
      />
      <TaskList
        filters={
          result === 'all' ? { miner: hotkey, kind } : { miner: hotkey, kind, verdict: result }
        }
        showMiner={false}
        emptyTitle={
          result === 'all'
            ? `This miner has no finalized ${kind} task yet.`
            : `No ${NONE_WITH[result]} ${kind} task.`
        }
      />
    </Card>
  )
}

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
      >
        <KindSwitch kind={kind} onChange={setKind} noun="tasks" />
      </PageHeader>
      <QueryState query={query}>
        {(miner) =>
          miner.known ? (
            <>
              {KINDS.map((each) => (
                <MinerNotice key={each} kind={each} pool={miner.pools[each]} />
              ))}
              <MinerStats miner={miner} kind={kind} />
              <WorkPanel kind={kind} miner={hotkey} />
              <ItsTasks hotkey={hotkey} kind={kind} />
              <div className="grid gap-4 xl:grid-cols-2">
                <Card>
                  <CardHeader
                    title="Take-backs"
                    description="Passes taken back after a later check of this miner failed"
                  />
                  <TakeBacks takeBacks={miner.take_backs} kind={kind} />
                </Card>
                <Card>
                  <CardHeader
                    title="Budget cuts"
                    description="A failed upload, a claim that ran out or a task given back halves the budget"
                  />
                  <BudgetCuts transitions={miner.transitions} kind={kind} />
                </Card>
              </div>
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
