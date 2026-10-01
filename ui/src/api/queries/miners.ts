import { useQuery } from '@tanstack/react-query'
import { apiGet } from '../client'
import { POLL_INTERVAL_MS, WINDOW_HOURS } from '../config'
import type { MinerDetail, MinersResponse } from '../types'

export function fetchMiner(hotkey: string, signal?: AbortSignal): Promise<MinerDetail> {
  return apiGet<MinerDetail>(`/v1/miners/${encodeURIComponent(hotkey)}`, {}, signal)
}

export function useMiners() {
  return useQuery({
    queryKey: ['miners', WINDOW_HOURS],
    queryFn: ({ signal }) => apiGet<MinersResponse>('/v1/miners', { hours: WINDOW_HOURS }, signal),
    refetchInterval: POLL_INTERVAL_MS,
  })
}

export function useMiner(hotkey: string) {
  return useQuery({
    queryKey: ['miner', hotkey],
    queryFn: ({ signal }) => fetchMiner(hotkey, signal),
    refetchInterval: POLL_INTERVAL_MS,
  })
}
