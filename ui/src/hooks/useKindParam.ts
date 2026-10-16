import { useSearchParams } from 'react-router'
import type { Kind } from '@/api/types'
import { DEFAULT_KIND, KIND_PARAM, readKind } from '@/lib/kinds'

export function useKindParam(): [Kind, (kind: Kind) => void] {
  const [params, setParams] = useSearchParams()
  const kind = readKind(params.get(KIND_PARAM)) ?? DEFAULT_KIND
  const setKind = (next: Kind) => {
    setParams(
      (current) => {
        const updated = new URLSearchParams(current)
        if (next === DEFAULT_KIND) updated.delete(KIND_PARAM)
        else updated.set(KIND_PARAM, next)
        return updated
      },
      { replace: true },
    )
  }
  return [kind, setKind]
}
