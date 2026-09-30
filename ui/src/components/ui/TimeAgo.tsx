import { EMPTY_VALUE, formatAbsolute, formatRelative } from '@/lib/format'
import { useNow } from '@/hooks/useNow'

export function TimeAgo({ at }: { at: number | null | undefined }) {
  const now = useNow()
  if (at === null || at === undefined) return <span className="text-ink-faint">{EMPTY_VALUE}</span>
  return (
    <time
      dateTime={new Date(at * 1000).toISOString()}
      title={formatAbsolute(at)}
      className="cursor-help whitespace-nowrap"
    >
      {formatRelative(at, now)}
    </time>
  )
}
