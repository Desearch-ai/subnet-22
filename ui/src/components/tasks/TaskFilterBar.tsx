import type { SubmitEvent } from 'react'
import type { TaskFilters } from '@/api/queries/tasks'
import { Button } from '@/components/ui/Button'
import { Input, Select } from '@/components/ui/Input'
import { VERDICT_LABEL, VERDICTS } from '@/lib/labels'

export const FILTER_MINER = 'miner'
export const FILTER_VALIDATOR = 'validator'
export const FILTER_VERDICT = 'verdict'

interface TaskFilterBarProps {
  filters: TaskFilters
  onApply: (form: FormData) => void
  onClear: () => void
}

export function TaskFilterBar({ filters, onApply, onClear }: TaskFilterBarProps) {
  const submit = (event: SubmitEvent<HTMLFormElement>) => {
    event.preventDefault()
    onApply(new FormData(event.currentTarget))
  }
  const hasFilters = Object.keys(filters).length > 0

  return (
    <form
      key={JSON.stringify(filters)}
      onSubmit={submit}
      className="border-line grid gap-2 border-b px-4 py-3 sm:grid-cols-2 lg:grid-cols-[1fr_1fr_10rem_auto]"
    >
      <Input
        name={FILTER_MINER}
        defaultValue={filters.miner ?? ''}
        placeholder="Miner hotkey"
        aria-label="Filter by miner hotkey"
        spellCheck={false}
        autoComplete="off"
      />
      <Input
        name={FILTER_VALIDATOR}
        defaultValue={filters.validator ?? ''}
        placeholder="Validator hotkey"
        aria-label="Filter by validator hotkey"
        spellCheck={false}
        autoComplete="off"
      />
      <Select
        name={FILTER_VERDICT}
        defaultValue={filters.verdict ?? ''}
        aria-label="Filter by result"
      >
        <option value="">Any result</option>
        {VERDICTS.map((verdict) => (
          <option key={verdict} value={verdict}>
            {VERDICT_LABEL[verdict]}
          </option>
        ))}
      </Select>
      <div className="flex gap-2">
        <Button type="submit" variant="primary">
          Apply
        </Button>
        {hasFilters ? <Button onClick={onClear}>Clear</Button> : null}
      </div>
    </form>
  )
}
