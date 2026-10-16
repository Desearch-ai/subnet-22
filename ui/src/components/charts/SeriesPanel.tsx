import type { ReactNode } from 'react'
import { useSeries } from '@/api/queries/series'
import type { Kind } from '@/api/types'
import { Card, CardHeader } from '@/components/ui/Card'
import { QueryState } from '@/components/ui/QueryState'
import { SeriesChart } from './SeriesChart'
import { KIND_SERIES } from './series'

const PERIOD = 'Per hour, last 24 hours'

interface SeriesPanelProps {
  kind: Kind
  miner?: string
  summary?: string | undefined
  actions?: ReactNode
}

export function SeriesPanel({ kind, miner, summary, actions }: SeriesPanelProps) {
  const query = useSeries(miner === undefined ? { kind } : { kind, miner })
  const chart = KIND_SERIES[kind]
  return (
    <Card>
      <CardHeader
        title={chart.title}
        description={summary ? `${PERIOD} · ${summary}` : PERIOD}
        actions={actions}
      />
      <QueryState query={query}>
        {(series) => (
          <div className="p-4">
            <SeriesChart
              title={chart.caption}
              points={series.points}
              series={chart.series}
              emptyTitle={chart.emptyTitle}
            />
          </div>
        )}
      </QueryState>
    </Card>
  )
}
