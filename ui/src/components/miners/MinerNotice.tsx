import type { Kind, MinerPool } from '@/api/types'
import { useNow } from '@/hooks/useNow'
import { formatAbsolute, formatInt } from '@/lib/format'
import { KIND_LABEL, KIND_WORDS } from '@/lib/kinds'

const WAITING_PER_BUDGET = 2

function notice(kind: Kind, pool: MinerPool, now: number): string | null {
  if (pool.locked_until !== null && pool.locked_until * 1000 > now) {
    return `Locked out: no new ${kind} tasks until ${formatAbsolute(pool.locked_until)}.`
  }
  if (pool.in_flight >= pool.budget) {
    return `${KIND_LABEL[kind]}: ${KIND_WORDS[kind].working} ${formatInt(pool.in_flight)} tasks, as many as its budget allows. It gets more as it uploads them.`
  }
  if (pool.waiting >= WAITING_PER_BUDGET * pool.budget) {
    return `${KIND_LABEL[kind]}: ${formatInt(pool.waiting)} uploads are waiting for a verdict, the most its budget allows. New tasks resume as verdicts arrive.`
  }
  return null
}

export function MinerNotice({ kind, pool }: { kind: Kind; pool: MinerPool }) {
  const text = notice(kind, pool, useNow())
  if (text === null) return null
  return (
    <p
      role="status"
      className="border-warn/40 bg-warn-wash text-ink rounded-lg border px-4 py-2.5 text-sm"
    >
      {text}
    </p>
  )
}
