import { cn } from '@/lib/cn'
import type { SortState } from '@/lib/sort'

const ARROW_ASCENDING = '↑'
const ARROW_DESCENDING = '↓'

interface SortThProps<Key extends string> {
  sortKey: Key
  sort: SortState<Key>
  onSort: (key: Key) => void
  label: string
  hint?: string
  align?: 'left' | 'right'
}

export function SortTh<Key extends string>({
  sortKey,
  sort,
  onSort,
  label,
  hint,
  align = 'right',
}: SortThProps<Key>) {
  const active = sort.key === sortKey
  const ascending = active && sort.direction === 'asc'
  return (
    <th
      scope="col"
      aria-sort={active ? (ascending ? 'ascending' : 'descending') : 'none'}
      className={cn(
        'border-line h-9 border-b px-3 whitespace-nowrap',
        align === 'right' ? 'text-right' : 'text-left',
      )}
    >
      <button
        type="button"
        title={hint}
        onClick={() => {
          onSort(sortKey)
        }}
        className={cn(
          'text-label hover:text-ink font-mono font-normal tracking-wider uppercase transition-colors',
          active ? 'text-ink' : 'text-ink-faint',
        )}
      >
        {label}
        <span aria-hidden className="ml-1 inline-block w-2">
          {active ? (ascending ? ARROW_ASCENDING : ARROW_DESCENDING) : ''}
        </span>
      </button>
    </th>
  )
}
