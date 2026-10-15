import { useQuery } from '@tanstack/react-query'
import { apiGet } from '../client'
import { POLL_INTERVAL_MS, WINDOW_HOURS } from '../config'
import type { ValidatorDetail, ValidatorsResponse } from '../types'
import { queryKeys } from './keys'

export function fetchValidator(hotkey: string, signal?: AbortSignal): Promise<ValidatorDetail> {
  return apiGet<ValidatorDetail>(
    `/v1/validators/${encodeURIComponent(hotkey)}`,
    { hours: WINDOW_HOURS },
    signal,
  )
}

export function useValidators() {
  return useQuery({
    queryKey: queryKeys.validators(),
    queryFn: ({ signal }) =>
      apiGet<ValidatorsResponse>('/v1/validators', { hours: WINDOW_HOURS }, signal),
    refetchInterval: POLL_INTERVAL_MS,
  })
}

export function useValidator(hotkey: string) {
  return useQuery({
    queryKey: queryKeys.validator(hotkey),
    queryFn: ({ signal }) => fetchValidator(hotkey, signal),
    refetchInterval: POLL_INTERVAL_MS,
  })
}
