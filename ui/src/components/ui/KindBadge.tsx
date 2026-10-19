import type { Kind } from '@/api/types'
import { KIND_LABEL, KIND_SUMMARY } from '@/lib/kinds'
import { Badge } from './Badge'

export function KindBadge({ kind }: { kind: Kind }) {
  return <Badge title={`${KIND_LABEL[kind]}: ${KIND_SUMMARY[kind]}`}>{KIND_LABEL[kind]}</Badge>
}
