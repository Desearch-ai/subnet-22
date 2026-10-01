export type SortDirection = 'asc' | 'desc'

export interface SortState<Key extends string> {
  key: Key
  direction: SortDirection
}

export type SortValue = number | string | boolean | null

export function compareValues(left: SortValue, right: SortValue): number {
  if (left === right) return 0
  if (left === null) return -1
  if (right === null) return 1
  if (typeof left === 'string' && typeof right === 'string') return left.localeCompare(right)
  return Number(left) - Number(right)
}

export function sortRows<Row, Key extends string>(
  rows: readonly Row[],
  state: SortState<Key>,
  valueOf: (row: Row, key: Key) => SortValue,
): Row[] {
  const sign = state.direction === 'asc' ? 1 : -1
  return [...rows].sort(
    (left, right) => sign * compareValues(valueOf(left, state.key), valueOf(right, state.key)),
  )
}

export function nextSort<Key extends string>(current: SortState<Key>, key: Key): SortState<Key> {
  if (current.key !== key) return { key, direction: 'desc' }
  return { key, direction: current.direction === 'desc' ? 'asc' : 'desc' }
}
