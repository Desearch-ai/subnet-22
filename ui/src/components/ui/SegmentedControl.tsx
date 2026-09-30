import { cn } from '@/lib/cn'

export interface Segment<Value extends string | number> {
  value: Value
  label: string
}

interface SegmentedControlProps<Value extends string | number> {
  label: string
  segments: readonly Segment<Value>[]
  value: Value
  onChange: (value: Value) => void
}

export function SegmentedControl<Value extends string | number>({
  label,
  segments,
  value,
  onChange,
}: SegmentedControlProps<Value>) {
  return (
    <div
      role="group"
      aria-label={label}
      className="border-line-strong scroll-thin inline-flex max-w-full overflow-x-auto rounded-md border"
    >
      {segments.map((segment) => (
        <button
          key={segment.value}
          type="button"
          aria-pressed={segment.value === value}
          onClick={() => {
            onChange(segment.value)
          }}
          className={cn(
            'border-line-strong h-7 shrink-0 border-l px-2.5 font-mono text-xs whitespace-nowrap transition-colors first:border-l-0',
            segment.value === value
              ? 'bg-raised text-ink'
              : 'text-ink-muted hover:bg-raised hover:text-ink',
          )}
        >
          {segment.label}
        </button>
      ))}
    </div>
  )
}
