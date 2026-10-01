import { Badge } from '@/components/ui/Badge'
import { useNow } from '@/hooks/useNow'
import { EMPTY_VALUE, formatAbsolute, formatRelative } from '@/lib/format'

export function Lockout({ until }: { until: number | null }) {
  const now = useNow()
  if (until === null || until * 1000 <= now) {
    return <span className="text-ink-faint">{EMPTY_VALUE}</span>
  }
  return (
    <Badge tone="warn" title={`No new tasks until ${formatAbsolute(until)}`}>
      ends {formatRelative(until, now)}
    </Badge>
  )
}
