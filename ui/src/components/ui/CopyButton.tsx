import { useEffect, useState } from 'react'

const COPIED_NOTICE_MS = 1_500
const ICON_COPY = 'M5 5V2.5h8.5V11H11M2.5 5H11v8.5H2.5z'
const ICON_DONE = 'M3 8.5l3.2 3.2L13 4.8'

export function CopyButton({ value, label }: { value: string; label: string }) {
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    if (!copied) return
    const timer = setTimeout(() => {
      setCopied(false)
    }, COPIED_NOTICE_MS)
    return () => {
      clearTimeout(timer)
    }
  }, [copied])

  const copy = () => {
    void navigator.clipboard.writeText(value).then(() => {
      setCopied(true)
    })
  }

  return (
    <button
      type="button"
      onClick={copy}
      aria-label={copied ? `Copied ${label}` : `Copy ${label}`}
      title={copied ? 'Copied' : `Copy ${label}`}
      className="text-ink-faint hover:text-ink shrink-0 rounded-sm p-0.5 transition-colors"
    >
      <svg
        aria-hidden
        viewBox="0 0 16 16"
        className="size-3"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d={copied ? ICON_DONE : ICON_COPY} />
      </svg>
    </button>
  )
}
