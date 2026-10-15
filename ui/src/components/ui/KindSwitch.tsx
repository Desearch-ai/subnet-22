import type { Kind } from '@/api/types'
import { KIND_LABEL, KINDS } from '@/lib/kinds'
import { SegmentedControl } from './SegmentedControl'

interface KindSwitchProps {
  kind: Kind
  onChange: (kind: Kind) => void
  noun?: string
}

export function KindSwitch({ kind, onChange, noun }: KindSwitchProps) {
  const segments = KINDS.map((each) => ({
    value: each,
    label: noun ? `${KIND_LABEL[each]} ${noun}` : KIND_LABEL[each],
  }))
  return (
    <SegmentedControl label="Kind of task" segments={segments} value={kind} onChange={onChange} />
  )
}
