import type { ReactNode } from 'react'

export function FieldList({ children }: { children: ReactNode }) {
  return <dl className="divide-line divide-y text-sm">{children}</dl>
}

interface FieldProps {
  label: string
  hint?: string
  children: ReactNode
}

export function Field({ label, hint, children }: FieldProps) {
  return (
    <div className="grid grid-cols-[8.5rem_minmax(0,1fr)] items-center gap-3 px-4 py-2">
      <dt className="text-ink-muted" title={hint}>
        {label}
      </dt>
      <dd className="min-w-0">{children}</dd>
    </div>
  )
}
