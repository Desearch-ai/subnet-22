import type { ReactNode } from 'react'
import { cn } from '@/lib/cn'

interface StatProps {
  label: string
  value: ReactNode
  detail?: ReactNode
  className?: string
}

export function Stat({ label, value, detail, className }: StatProps) {
  return (
    <div className={cn('border-line bg-surface min-w-0 rounded-lg border px-4 py-3', className)}>
      <div className="text-ink-faint text-label font-mono tracking-wider uppercase">{label}</div>
      <div className="text-ink mt-1 overflow-x-auto font-mono text-xl font-medium whitespace-nowrap tabular-nums">
        {value}
      </div>
      {detail ? <div className="text-ink-muted mt-0.5 text-xs">{detail}</div> : null}
    </div>
  )
}

export function StatGrid({ children }: { children: ReactNode }) {
  return <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-5">{children}</div>
}
