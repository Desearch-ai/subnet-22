import { useQuery } from '@tanstack/react-query'
import { apiGet } from '../client'
import { HOURS_SHOWN, POLL_INTERVAL_MS } from '../config'
import type { Hours } from '../types'
import { queryKeys, type HoursFilters } from './keys'

export function useHours(filters: HoursFilters) {
  return useQuery({
    queryKey: queryKeys.hours(filters),
    queryFn: ({ signal }) =>
      apiGet<Hours>('/v1/stats/hours', { hours: HOURS_SHOWN, ...filters }, signal),
    refetchInterval: POLL_INTERVAL_MS,
  })
}
