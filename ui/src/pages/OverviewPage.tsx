import { useOverview } from '@/api/queries/overview'
import { LivePanel } from '@/components/overview/LivePanel'
import { OverviewStats } from '@/components/overview/OverviewStats'
import { RecentTasks } from '@/components/overview/RecentTasks'
import { TopMiners } from '@/components/overview/TopMiners'
import { Card } from '@/components/ui/Card'
import { PageHeader } from '@/components/ui/PageHeader'
import { QueryState } from '@/components/ui/QueryState'
import { TimeAgo } from '@/components/ui/TimeAgo'

export function OverviewPage() {
  const overview = useOverview()
  return (
    <div className="space-y-4">
      <PageHeader
        title="Overview"
        description="Miners claim batches of URLs to crawl. Validators check a sample of each upload and vote. The rows that hold up count toward the miner's share of emission."
      >
        {overview.data ? (
          <span className="text-ink-faint text-xs">
            updated <TimeAgo at={overview.data.as_of} />
          </span>
        ) : null}
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
        <RecentTasks />
        <TopMiners />
      </div>
    </div>
  )
}
