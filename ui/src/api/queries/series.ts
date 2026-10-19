import { useQuery } from '@tanstack/react-query'
import { apiGet } from '../client'
import { POLL_INTERVAL_MS } from '../config'
import type { Series } from '../types'
import { queryKeys, type SeriesFilters } from './keys'

const BUCKET_MINUTES = 60
const BUCKETS = 24

export function useSeries(filters: SeriesFilters) {
  return useQuery({
    queryKey: queryKeys.series(filters),
    queryFn: ({ signal }) =>
      apiGet<Series>(
        '/v1/stats/series',
        { bucket_minutes: BUCKET_MINUTES, buckets: BUCKETS, ...filters },
        signal,
      ),
    refetchInterval: POLL_INTERVAL_MS,
  })
}
