import { useQuery } from '@tanstack/react-query'
import { apiGet } from '../client'
import { POLL_INTERVAL_MS } from '../config'
import type { Series } from '../types'

const BUCKET_MINUTES = 60
const BUCKETS = 24

export function useMinerSeries(miner: string) {
  return useQuery({
    queryKey: ['series', miner],
    queryFn: ({ signal }) =>
      apiGet<Series>(
        '/v1/stats/series',
        { bucket_minutes: BUCKET_MINUTES, buckets: BUCKETS, miner },
        signal,
      ),
    refetchInterval: POLL_INTERVAL_MS,
  })
}
