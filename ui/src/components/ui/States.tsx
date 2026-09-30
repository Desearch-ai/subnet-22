import type { ReactNode } from 'react'
import { ApiError, UnreachableError } from '@/api/client'
import { API_BASE_URL } from '@/api/config'
import { Button } from './Button'

export function LoadingState({ label = 'Loading' }: { label?: string }) {
  return (
    <div role="status" className="text-ink-muted flex items-center gap-2 px-4 py-8 text-sm">
      <span aria-hidden className="bg-ink-faint size-1.5 animate-pulse rounded-full" />
      {label}…
    </div>
  )
}

interface EmptyStateProps {
  title: string
  children?: ReactNode
}

export function EmptyState({ title, children }: EmptyStateProps) {
  return (
    <div className="px-4 py-8 text-sm">
      <p className="text-ink">{title}</p>
      {children ? <p className="text-ink-muted mt-1">{children}</p> : null}
    </div>
  )
}

function describeError(error: Error): { title: string; detail: ReactNode } {
  if (error instanceof UnreachableError) {
    return {
      title: 'The task API cannot be reached',
      detail: (
        <>
          No answer from <span className="text-ink font-mono text-xs">{API_BASE_URL}</span>. Check
          that the API is running and that it allows requests from this site.
        </>
      ),
    }
  }
  if (error instanceof ApiError && error.isBackoff) {
    return {
      title: 'The task API is busy',
      detail: 'It asked this page to slow down. The data loads again shortly.',
    }
  }
  if (error instanceof ApiError) {
    return {
      title: `The task API answered with an error (${error.status})`,
      detail: (
        <>
          Request to <span className="text-ink font-mono text-xs">{API_BASE_URL}</span> failed.
        </>
      ),
    }
  }
  return { title: 'Something went wrong', detail: error.message }
}

interface ErrorStateProps {
  error: Error
  onRetry?: () => void
}

export function ErrorState({ error, onRetry }: ErrorStateProps) {
  const { title, detail } = describeError(error)
  return (
    <div role="alert" className="px-4 py-8 text-sm">
      <p className="text-ink flex items-center gap-2">
        <span aria-hidden className="bg-fail size-1.5 rounded-full" />
        {title}
      </p>
      <p className="text-ink-muted mt-1">{detail}</p>
      {onRetry ? (
        <Button onClick={onRetry} className="mt-3">
          Try again
        </Button>
      ) : null}
    </div>
  )
}

export function StaleNotice() {
  return (
    <p role="status" className="border-line text-ink-muted border-b px-4 py-1.5 text-xs">
      The last refresh failed. Showing the most recent data received.
    </p>
  )
}
