import { useMinerSeries } from '@/api/queries/series'
import { SeriesChart } from '@/components/charts/SeriesChart'
import { ROW_SERIES } from '@/components/charts/series'
import { Card, CardHeader } from '@/components/ui/Card'
import { QueryState } from '@/components/ui/QueryState'

export function MinerRowsPanel({ miner }: { miner: string }) {
  const query = useMinerSeries(miner)
  return (
    <Card>
      <CardHeader title="Rows over time" description="Per hour, last 24 hours" />
      <QueryState query={query}>
        {(series) => (
          <div className="p-4">
            <SeriesChart
              title="Rows returned and counted"
              points={series.points}
              series={ROW_SERIES}
              emptyTitle="No rows were returned in this period."
            />
          </div>
        )}
      </QueryState>
    </Card>
  )
}
