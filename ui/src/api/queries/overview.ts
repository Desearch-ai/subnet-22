import { useQuery } from '@tanstack/react-query'
import { apiGet } from '../client'
import { POLL_INTERVAL_MS, WINDOW_HOURS } from '../config'
import type { Live, Overview } from '../types'

export function useOverview() {
  return useQuery({
    queryKey: ['overview', WINDOW_HOURS],
    queryFn: ({ signal }) => apiGet<Overview>('/v1/overview', { hours: WINDOW_HOURS }, signal),
    refetchInterval: POLL_INTERVAL_MS,
  })
}

export function useLive() {
  return useQuery({
    queryKey: ['live'],
    queryFn: ({ signal }) => apiGet<Live>('/v1/live', {}, signal),
    refetchInterval: POLL_INTERVAL_MS,
  })
}
