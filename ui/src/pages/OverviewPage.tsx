import { useOverview } from '@/api/queries/overview'
import type { Kind, Overview } from '@/api/types'
import { SeriesPanel } from '@/components/charts/SeriesPanel'
import { LivePanel } from '@/components/overview/LivePanel'
import { OverviewStats } from '@/components/overview/OverviewStats'
import { RecentTasks } from '@/components/overview/RecentTasks'
import { TopMiners } from '@/components/overview/TopMiners'
import { Card } from '@/components/ui/Card'
import { KindSwitch } from '@/components/ui/KindSwitch'
import { PageHeader } from '@/components/ui/PageHeader'
import { QueryState } from '@/components/ui/QueryState'
import { TimeAgo } from '@/components/ui/TimeAgo'
import { useKindParam } from '@/hooks/useKindParam'
import { formatInt } from '@/lib/format'

function windowSummary(kind: Kind, { window }: Overview): string {
  return kind === 'embed'
    ? `${formatInt(window.tasks)} tasks, ${formatInt(window.credited)} characters counted`
    : `${formatInt(window.returned)} rows returned, ${formatInt(window.credited)} counted`
}

export function OverviewPage() {
  const [kind, setKind] = useKindParam()
  const overview = useOverview(kind)
  const summary = overview.data ? windowSummary(kind, overview.data) : undefined
  return (
    <div className="space-y-4">
      <PageHeader
        title="Overview"
        description="Miners claim two kinds of task. A crawl task is up to 1,000 URLs, returned as one row per URL. An embed task is a batch of page texts, returned as one vector per text. Validators check a sample of each upload and vote, and what holds up counts toward the miner's share of that kind's pool."
      >
        <div className="flex flex-wrap items-center gap-3">
          {overview.data ? (
            <span className="text-ink-faint text-xs">
              updated <TimeAgo at={overview.data.as_of} />
            </span>
          ) : null}
          <KindSwitch kind={kind} onChange={setKind} />
        </div>
      </PageHeader>
      {overview.data ? (
        <OverviewStats overview={overview.data} />
      ) : (
        <Card>
          <QueryState query={overview}>{() => null}</QueryState>
        </Card>
      )}
      <LivePanel />
      <div className="grid gap-4 xl:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
        <SeriesPanel kind={kind} summary={summary} />
        <TopMiners kind={kind} />
      </div>
      <RecentTasks />
    </div>
  )
}
