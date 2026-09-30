import type { ReactNode, TdHTMLAttributes } from 'react'
import { cn } from '@/lib/cn'

type Align = 'left' | 'right'

const ALIGN_CLASS: Record<Align, string> = {
  left: 'text-left',
  right: 'text-right',
}

export function Table({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className="scroll-thin w-full overflow-x-auto">
      <table className={cn('w-full border-collapse text-sm', className)}>{children}</table>
    </div>
  )
}

interface ThProps {
  children?: ReactNode
  align?: Align
  hint?: string
  className?: string
}

export function Th({ children, align = 'left', hint, className }: ThProps) {
  return (
    <th
      scope="col"
      title={hint}
      className={cn(
        'border-line text-ink-faint text-label h-9 border-b px-3 font-mono font-normal tracking-wider whitespace-nowrap uppercase',
        ALIGN_CLASS[align],
        hint ? 'cursor-help' : null,
        className,
      )}
    >
      <span
        className={
          hint ? 'decoration-line-strong underline decoration-dotted underline-offset-4' : undefined
        }
      >
        {children}
      </span>
    </th>
  )
}

interface TdProps extends TdHTMLAttributes<HTMLTableCellElement> {
  align?: Align
  numeric?: boolean
}

export function Td({ align, numeric = false, className, ...rest }: TdProps) {
  return (
    <td
      className={cn(
        'border-line border-b px-3 py-2 align-middle whitespace-nowrap',
        ALIGN_CLASS[align ?? (numeric ? 'right' : 'left')],
        numeric ? 'font-mono tabular-nums' : null,
        className,
      )}
      {...rest}
    />
  )
}

export function Tr({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <tr
      className={cn('hover:bg-raised/60 transition-colors [&:last-child>td]:border-b-0', className)}
    >
      {children}
    </tr>
  )
}
