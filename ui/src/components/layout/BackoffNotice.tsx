import { useSyncExternalStore } from 'react'
import { getPausedUntil, subscribeToPause } from '@/api/backoff'
import { TICK_SECOND_MS, useNow } from '@/hooks/useNow'

export function BackoffNotice() {
  const pausedUntil = useSyncExternalStore(subscribeToPause, getPausedUntil)
  const now = useNow(TICK_SECOND_MS)
  const secondsLeft = Math.ceil((pausedUntil - now) / 1000)
  if (secondsLeft <= 0) return null
  return (
    <div
      role="status"
      className="border-warn/40 bg-warn-wash text-ink-muted border-b px-4 py-1.5 text-center text-xs"
    >
      The task API asked this page to slow down. Refreshing again in {secondsLeft} s.
    </div>
  )
}
