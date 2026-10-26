import { Link } from 'react-router'
import { cn } from '@/lib/cn'
import { EMPTY_VALUE, truncateMiddle } from '@/lib/format'
import { CopyButton } from './CopyButton'

interface NeuronProps {
  hotkey: string | null
  label: string
  uid: number | null
  coldkey?: string | null
  to?: string
  className?: string
}

export function Neuron({ hotkey, label, uid, coldkey = null, to, className }: NeuronProps) {
  if (!hotkey) return <span className="text-ink-faint font-mono text-xs">{EMPTY_VALUE}</span>
  return (
    <span className={cn('inline-flex max-w-full min-w-0 flex-col font-mono text-xs', className)}>
      <span className="inline-flex min-w-0 items-center gap-1.5">
        {uid === null ? null : <span className="text-ink shrink-0">UID {uid}</span>}
        {to ? (
          <Link to={to} title={hotkey} className="text-accent whitespace-nowrap hover:underline">
            {truncateMiddle(hotkey)}
          </Link>
        ) : (
          <span title={hotkey} className="text-ink-muted whitespace-nowrap">
            {truncateMiddle(hotkey)}
          </span>
        )}
        <CopyButton value={hotkey} label={label} />
      </span>
      {coldkey ? (
        <span className="text-ink-faint inline-flex min-w-0 items-center gap-1 text-[11px]">
          <span className="whitespace-nowrap" title={coldkey}>
            coldkey {truncateMiddle(coldkey)}
          </span>
          <CopyButton value={coldkey} label="coldkey" />
        </span>
      ) : null}
    </span>
  )
}
