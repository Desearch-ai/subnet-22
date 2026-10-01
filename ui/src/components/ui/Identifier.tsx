import { Link } from 'react-router'
import { cn } from '@/lib/cn'
import { truncateMiddle } from '@/lib/format'
import { CopyButton } from './CopyButton'

interface IdentifierProps {
  value: string
  label: string
  to?: string
  uid?: number | null
  full?: boolean
  className?: string
}

export function Identifier({
  value,
  label,
  to,
  uid = null,
  full = false,
  className,
}: IdentifierProps) {
  const text = full ? value : uid === null ? truncateMiddle(value) : `UID ${uid}`
  return (
    <span
      className={cn(
        'inline-flex max-w-full min-w-0 items-center gap-1 font-mono text-xs',
        className,
      )}
    >
      {to ? (
        <Link
          to={to}
          title={value}
          className={cn('text-accent hover:underline', full ? 'break-all' : 'whitespace-nowrap')}
        >
          {text}
        </Link>
      ) : (
        <span title={value} className={full ? 'break-all' : 'whitespace-nowrap'}>
          {text}
        </span>
      )}
      <CopyButton value={value} label={label} />
    </span>
  )
}
