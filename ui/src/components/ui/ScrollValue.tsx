import type { ReactNode } from 'react'
import { cn } from '@/lib/cn'

export function ScrollValue({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cn('scroll-thin max-w-full overflow-x-auto', className)}>
      <span className="font-mono text-xs whitespace-nowrap">{children}</span>
    </div>
  )
}
