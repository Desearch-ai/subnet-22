import { useQuery } from '@tanstack/react-query'
import { apiGet } from '../client'
import { POLL_INTERVAL_MS, WINDOW_HOURS } from '../config'
import type { Kind, MinerDetail, MinersResponse } from '../types'
import { queryKeys } from './keys'

export function fetchMiner(hotkey: string, signal?: AbortSignal): Promise<MinerDetail> {
  return apiGet<MinerDetail>(`/v1/miners/${encodeURIComponent(hotkey)}`, {}, signal)
}

export function useMiners(kind: Kind) {
  return useQuery({
    queryKey: queryKeys.miners(kind),
    queryFn: ({ signal }) =>
      apiGet<MinersResponse>('/v1/miners', { hours: WINDOW_HOURS, kind }, signal),
    refetchInterval: POLL_INTERVAL_MS,
  })
}

export function useMiner(hotkey: string) {
  return useQuery({
    queryKey: queryKeys.miner(hotkey),
    queryFn: ({ signal }) => fetchMiner(hotkey, signal),
    refetchInterval: POLL_INTERVAL_MS,
  })
}
