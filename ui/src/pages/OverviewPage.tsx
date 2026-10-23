import { useHours } from '@/api/queries/hours'
import { useOverview } from '@/api/queries/overview'
import { WorkPanel } from '@/components/charts/WorkPanel'
import { ChecksPanel } from '@/components/overview/ChecksPanel'
import { LivePanel } from '@/components/overview/LivePanel'
import { LiveStats } from '@/components/overview/LiveStats'
import { MinerLookup } from '@/components/overview/MinerLookup'
import { RecentProblems } from '@/components/overview/RecentProblems'
import { TopMiners } from '@/components/overview/TopMiners'
import { ValidatorsPanel } from '@/components/overview/ValidatorsPanel'
import { Card } from '@/components/ui/Card'
import { KindSwitch } from '@/components/ui/KindSwitch'
import { PageHeader } from '@/components/ui/PageHeader'
import { QueryState } from '@/components/ui/QueryState'
import { TimeAgo } from '@/components/ui/TimeAgo'
import { useKindParam } from '@/hooks/useKindParam'

export function OverviewPage() {
  const crawl = useOverview('crawl')
  const embed = useOverview('embed')
  const [kind, setKind] = useKindParam()
  const hours = useHours({ kind })
  const chosen = kind === 'embed' ? embed : crawl
  return (
    <div className="space-y-4">
      <PageHeader
        title="Overview"
        description="How much work there is, how fast uploads are checked and counted, and what fails and why. A crawl task is up to 1,000 URLs returned as one row each; an embed task is a batch of page texts returned as one vector each."
      >
        {crawl.data ? (
          <span className="text-ink-faint text-xs">
            updated <TimeAgo at={crawl.data.as_of} />
          </span>
        ) : null}
      </PageHeader>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <KindSwitch kind={kind} onChange={setKind} noun="tasks" />
        <MinerLookup />
      </div>
      {crawl.data && embed.data ? (
        <LiveStats
          overviews={{ crawl: crawl.data, embed: embed.data }}
          kind={kind}
          hours={hours.data}
        />
      ) : (
        <Card>
          <QueryState query={crawl.data ? embed : crawl}>{() => null}</QueryState>
        </Card>
      )}
      <WorkPanel kind={kind} />
      <QueryState query={chosen}>{(overview) => <ChecksPanel overview={overview} />}</QueryState>
      <RecentProblems kind={kind} />
      <div className="grid gap-4 xl:grid-cols-2">
        <TopMiners kind={kind} />
        <ValidatorsPanel />
      </div>
      <LivePanel />
    </div>
  )
}
