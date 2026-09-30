import { Identifier } from './Identifier'

interface NeuronKeysProps {
  hotkey: string
  label: string
  uid: number | null
  coldkey: string | null
}

export function NeuronKeys({ hotkey, label, uid, coldkey }: NeuronKeysProps) {
  return (
    <span className="flex flex-col gap-0.5">
      {uid === null ? null : <span className="text-ink font-mono text-sm">UID {uid}</span>}
      <span className="flex flex-wrap items-center gap-x-2">
        <span className="text-ink-faint text-xs">hotkey</span>
        <Identifier value={hotkey} label={label} full />
      </span>
      {coldkey === null ? null : (
        <span className="flex flex-wrap items-center gap-x-2">
          <span className="text-ink-faint text-xs">coldkey</span>
          <Identifier value={coldkey} label="coldkey" full />
        </span>
      )}
    </span>
  )
}
