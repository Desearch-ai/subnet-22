import { QueryClient } from '@tanstack/react-query'
import { ApiError } from './client'
import { MAX_RETRIES, POLL_INTERVAL_MS } from './config'

const RETRY_DELAY_MS = 1_000

function shouldRetry(failureCount: number, error: Error): boolean {
  if (error instanceof ApiError && error.isNotFound) return false
  return failureCount < MAX_RETRIES
}

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: POLL_INTERVAL_MS,
      retry: shouldRetry,
      retryDelay: RETRY_DELAY_MS,
      refetchIntervalInBackground: false,
      refetchOnWindowFocus: true,
    },
  },
})
