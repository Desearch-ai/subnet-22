import { useQuery } from '@tanstack/react-query'
import { apiGet } from '../client'
import { POLL_INTERVAL_MS, WINDOW_HOURS } from '../config'
import type { Kind, Overview } from '../types'
import { queryKeys } from './keys'

export function useOverview(kind: Kind) {
  return useQuery({
    queryKey: queryKeys.overview(kind),
    queryFn: ({ signal }) =>
      apiGet<Overview>('/v1/overview', { hours: WINDOW_HOURS, kind }, signal),
    refetchInterval: POLL_INTERVAL_MS,
  })
}
