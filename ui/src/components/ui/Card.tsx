import type { ReactNode } from 'react'
import { cn } from '@/lib/cn'

interface CardProps {
  children: ReactNode
  className?: string
}

export function Card({ children, className }: CardProps) {
  return (
    <section className={cn('border-line bg-surface min-w-0 rounded-lg border', className)}>
      {children}
    </section>
  )
}

interface CardHeaderProps {
  title: string
  description?: ReactNode
  actions?: ReactNode
}

export function CardHeader({ title, description, actions }: CardHeaderProps) {
  return (
    <header className="border-line flex flex-wrap items-center justify-between gap-x-4 gap-y-2 border-b px-4 py-3">
      <div className="min-w-0">
        <h2 className="text-ink text-sm font-medium">{title}</h2>
        {description ? <p className="text-ink-muted text-xs">{description}</p> : null}
      </div>
      {actions ? (
        <div className="flex max-w-full min-w-0 flex-wrap items-center gap-2">{actions}</div>
      ) : null}
    </header>
  )
}
