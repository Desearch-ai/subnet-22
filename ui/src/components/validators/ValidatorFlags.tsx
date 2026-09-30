import type { ValidatorSummary } from '@/api/types'
import { Badge } from '@/components/ui/Badge'
import { EXCLUDED_MEANING } from '@/lib/labels'

export function ValidatorFlags({ validator }: { validator: ValidatorSummary }) {
  return (
    <span className="inline-flex flex-wrap gap-1">
      {validator.active ? (
        <Badge tone="pass" title="Asked for work or voted recently">
          Active
        </Badge>
      ) : (
        <Badge title="Has not asked for work or voted recently">Inactive</Badge>
      )}
      {validator.excluded ? (
        <Badge tone="fail" title={EXCLUDED_MEANING}>
          Excluded · no work
        </Badge>
      ) : null}
    </span>
  )
}
