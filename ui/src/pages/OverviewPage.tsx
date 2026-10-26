import { useHours } from '@/api/queries/hours'
import { useOverview } from '@/api/queries/overview'
import { WorkPanel } from '@/components/charts/WorkPanel'
import { FailReasons } from '@/components/overview/FailReasons'
import { LiveStats } from '@/components/overview/LiveStats'
import { RecentProblems } from '@/components/overview/RecentProblems'
import { TopMiners } from '@/components/overview/TopMiners'
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
        description="How much work there is right now, how fast uploads get their verdict, and why some do not pass."
      >
        <div className="flex items-center gap-3">
          {crawl.data ? (
            <span className="text-ink-faint text-xs">
              updated <TimeAgo at={crawl.data.as_of} />
            </span>
          ) : null}
          <KindSwitch kind={kind} onChange={setKind} noun="tasks" />
        </div>
      </PageHeader>
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
      <div className="grid gap-4 xl:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
        <WorkPanel kind={kind} />
        <TopMiners kind={kind} />
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        <QueryState query={chosen}>{(overview) => <FailReasons overview={overview} />}</QueryState>
        <RecentProblems kind={kind} />
      </div>
    </div>
  )
}
