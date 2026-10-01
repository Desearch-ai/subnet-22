import type { ReactNode } from 'react'
import { cn } from '@/lib/cn'

export type Tone = 'neutral' | 'pass' | 'fail' | 'void' | 'warn' | 'accent'

const TONE_FRAME: Record<Tone, string> = {
  neutral: 'border-line-strong bg-raised',
  pass: 'border-pass/40 bg-pass-wash',
  fail: 'border-fail/40 bg-fail-wash',
  void: 'border-void/40 bg-void-wash',
  warn: 'border-warn/40 bg-warn-wash',
  accent: 'border-accent/40 bg-accent-wash',
}

const TONE_DOT: Record<Tone, string> = {
  neutral: 'bg-ink-faint',
  pass: 'bg-pass',
  fail: 'bg-fail',
  void: 'bg-void',
  warn: 'bg-warn',
  accent: 'bg-accent',
}

interface BadgeProps {
  tone?: Tone
  children: ReactNode
  title?: string
}

export function Badge({ tone = 'neutral', children, title }: BadgeProps) {
  return (
    <span
      title={title}
      className={cn(
        'text-ink text-label inline-flex items-center gap-1.5 rounded-md border px-1.5 py-0.5 font-mono whitespace-nowrap uppercase',
        title ? 'cursor-help' : null,
        TONE_FRAME[tone],
      )}
    >
      <Dot tone={tone} />
      {children}
    </span>
  )
}

export function Dot({ tone }: { tone: Tone }) {
  return <span aria-hidden className={cn('size-1.5 shrink-0 rounded-full', TONE_DOT[tone])} />
}
