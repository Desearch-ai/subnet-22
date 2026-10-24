import { useId, useState, type KeyboardEvent } from 'react'
import { useNavigate } from 'react-router'
import { useMiners } from '@/api/queries/miners'
import { cn } from '@/lib/cn'
import { EMPTY_VALUE, formatPercent, truncateMiddle } from '@/lib/format'
import { DEFAULT_KIND } from '@/lib/kinds'
import { minerPath } from '@/lib/paths'
import { findMiners, looksLikeHotkey } from '@/lib/search'

const SHOWN = 6

function SearchIcon() {
  return (
    <svg
      aria-hidden
      viewBox="0 0 16 16"
      fill="none"
      className="text-ink-faint pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2"
    >
      <circle cx="7" cy="7" r="4.75" stroke="currentColor" strokeWidth="1.5" />
      <path d="m10.5 10.5 3 3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  )
}

export function MinerSearch() {
  const navigate = useNavigate()
  const listId = useId()
  const [text, setText] = useState('')
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(0)
  const [wanted, setWanted] = useState(false)
  const miners = useMiners(DEFAULT_KIND, wanted)
  const typed = text.trim()
  const found = findMiners(miners.data?.miners ?? [], typed, SHOWN)
  const options =
    found.length === 0 && looksLikeHotkey(typed) ? [typed] : found.map((m) => m.hotkey)
  const shown = open && typed !== ''
  const optionId = (index: number) => `${listId}-${index}`

  const go = (hotkey: string) => {
    setText('')
    setOpen(false)
    void navigate(minerPath(hotkey))
  }
  const onKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault()
      setOpen(true)
      const step = event.key === 'ArrowDown' ? 1 : -1
      setActive((index) => Math.min(Math.max(index + step, 0), Math.max(options.length - 1, 0)))
    } else if (event.key === 'Enter') {
      const hotkey = options[active]
      if (hotkey) go(hotkey)
    } else if (event.key === 'Escape') {
      setOpen(false)
    }
  }

  return (
    <div className="relative w-full sm:ml-auto sm:w-72 sm:shrink-0">
      <SearchIcon />
      <input
        type="search"
        role="combobox"
        aria-label="Find a miner by UID or hotkey"
        aria-expanded={shown}
        aria-controls={listId}
        aria-autocomplete="list"
        aria-activedescendant={shown && options.length > 0 ? optionId(active) : undefined}
        placeholder="Find a miner: UID or hotkey"
        autoComplete="off"
        spellCheck={false}
        value={text}
        onChange={(event) => {
          setText(event.target.value)
          setActive(0)
          setOpen(true)
        }}
        onFocus={() => {
          setWanted(true)
          setOpen(true)
        }}
        onBlur={() => {
          setOpen(false)
        }}
        onKeyDown={onKeyDown}
        className="border-line-strong bg-ground text-ink placeholder:text-ink-faint h-8 w-full rounded-md border pr-2.5 pl-8 text-xs"
      />
      {shown ? (
        <ul
          id={listId}
          role="listbox"
          className="border-line-strong bg-surface absolute top-full right-0 left-0 z-40 mt-1 overflow-hidden rounded-md border shadow-lg"
        >
          {options.length === 0 ? (
            <li className="text-ink-muted px-3 py-2 text-xs">
              {miners.data ? 'No miner matches' : 'Loading miners…'}
            </li>
          ) : (
            options.map((hotkey, index) => {
              const miner = found[index]
              return (
                <li
                  key={hotkey}
                  id={optionId(index)}
                  role="option"
                  aria-selected={index === active}
                  onMouseDown={(event) => {
                    event.preventDefault()
                    go(hotkey)
                  }}
                  onMouseEnter={() => {
                    setActive(index)
                  }}
                  className={cn(
                    'flex cursor-pointer items-center gap-3 px-3 py-2 text-xs',
                    index === active ? 'bg-raised' : null,
                  )}
                >
                  <span className="text-ink w-16 shrink-0 font-mono">
                    {miner === undefined
                      ? 'Open'
                      : miner.uid === null
                        ? EMPTY_VALUE
                        : `UID ${miner.uid}`}
                  </span>
                  <span className="text-ink-muted min-w-0 flex-1 truncate font-mono">
                    {truncateMiddle(hotkey)}
                  </span>
                  {miner ? (
                    <span className="text-ink-faint shrink-0" title="Share of the crawl pool">
                      {formatPercent(miner.share)}
                    </span>
                  ) : null}
                </li>
              )
            })
          )}
        </ul>
      ) : null}
    </div>
  )
}
