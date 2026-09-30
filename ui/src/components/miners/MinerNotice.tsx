import type { MinerPool } from '@/api/types'
import { useNow } from '@/hooks/useNow'
import { formatAbsolute, formatInt } from '@/lib/format'

export function MinerNotice({ pool }: { pool: MinerPool }) {
  const now = useNow()
  const locked = pool.locked_until !== null && pool.locked_until * 1000 > now
  const full = pool.in_flight >= pool.budget
  if (!locked && !full) return null
  return (
    <p
      role="status"
      className="border-warn/40 bg-warn-wash text-ink rounded-lg border px-4 py-2.5 text-sm"
    >
      {locked && pool.locked_until !== null
        ? `Locked out after failed checks: no new tasks until ${formatAbsolute(pool.locked_until)}.`
        : `Holding ${formatInt(pool.in_flight)} tasks, as many as its budget allows. It gets another once one of them is finalized.`}
    </p>
  )
}
