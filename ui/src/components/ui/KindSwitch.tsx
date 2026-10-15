import type { Kind } from '@/api/types'
import { KIND_LABEL, KINDS } from '@/lib/kinds'
import { SegmentedControl } from './SegmentedControl'

const SEGMENTS = KINDS.map((kind) => ({ value: kind, label: KIND_LABEL[kind] }))

interface KindSwitchProps {
  kind: Kind
  onChange: (kind: Kind) => void
}

export function KindSwitch({ kind, onChange }: KindSwitchProps) {
  return (
    <SegmentedControl label="Kind of task" segments={SEGMENTS} value={kind} onChange={onChange} />
  )
}
