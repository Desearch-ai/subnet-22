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
  /** The whole hotkey and coldkey, where there is room. */
  full?: boolean
  className?: string
}

/** A miner or validator: its UID and hotkey, and the coldkey under them when the metagraph has it. */
export function Neuron({
  hotkey,
  label,
  uid,
  coldkey = null,
  to,
  full = false,
  className,
}: NeuronProps) {
  if (!hotkey) return <span className="text-ink-faint font-mono text-xs">{EMPTY_VALUE}</span>
  const shown = (key: string) => (full ? key : truncateMiddle(key))
  const wrap = full ? 'break-all' : 'whitespace-nowrap'
  return (
    <span className={cn('inline-flex max-w-full min-w-0 flex-col font-mono text-xs', className)}>
      <span className="inline-flex min-w-0 items-center gap-1.5">
        {uid === null ? null : <span className="text-ink shrink-0">UID {uid}</span>}
        {to ? (
          <Link to={to} title={hotkey} className={cn('text-accent hover:underline', wrap)}>
            {shown(hotkey)}
          </Link>
        ) : (
          <span title={hotkey} className={cn('text-ink-muted', wrap)}>
            {shown(hotkey)}
          </span>
        )}
        <CopyButton value={hotkey} label={label} />
      </span>
      {coldkey ? (
        <span className="text-ink-faint inline-flex min-w-0 items-center gap-1 text-[11px]">
          <span className={wrap} title={coldkey}>
            coldkey {shown(coldkey)}
          </span>
          <CopyButton value={coldkey} label="coldkey" />
        </span>
      ) : null}
    </span>
  )
}
