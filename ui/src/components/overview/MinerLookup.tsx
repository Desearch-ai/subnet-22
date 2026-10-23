import type { SubmitEvent } from 'react'
import { useNavigate } from 'react-router'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { minerPath } from '@/lib/paths'

const FIELD = 'hotkey'

export function MinerLookup() {
  const navigate = useNavigate()
  const submit = (event: SubmitEvent<HTMLFormElement>) => {
    event.preventDefault()
    const entered = new FormData(event.currentTarget).get(FIELD)
    const hotkey = typeof entered === 'string' ? entered.trim() : ''
    if (hotkey) void navigate(minerPath(hotkey))
  }
  return (
    <form onSubmit={submit} role="search" className="flex min-w-0 items-center gap-2">
      <Input
        name={FIELD}
        aria-label="Miner hotkey"
        placeholder="Your miner hotkey"
        autoComplete="off"
        spellCheck={false}
        className="w-full sm:w-80"
      />
      <Button type="submit">Open</Button>
    </form>
  )
}
