import type { ReactNode } from 'react'
import { usePageTitle } from '@/hooks/usePageTitle'

interface PageHeaderProps {
  title: string
  description?: ReactNode
  children?: ReactNode
}

export function PageHeader({ title, description, children }: PageHeaderProps) {
  usePageTitle(title)
  return (
    <header className="mb-4 flex flex-wrap items-end justify-between gap-3">
      <div className="min-w-0">
        <h1 className="text-ink text-xl font-medium tracking-tight">{title}</h1>
        {description ? (
          <p className="text-ink-muted mt-0.5 max-w-3xl text-sm">{description}</p>
        ) : null}
      </div>
      {children}
    </header>
  )
}
