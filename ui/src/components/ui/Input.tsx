import type { InputHTMLAttributes, SelectHTMLAttributes } from 'react'
import { cn } from '@/lib/cn'

const CONTROL_CLASS =
  'border-line-strong bg-ground text-ink placeholder:text-ink-faint h-8 min-w-0 rounded-md border px-2.5 font-mono text-xs'

export function Input({ className, ...rest }: InputHTMLAttributes<HTMLInputElement>) {
  return <input className={cn(CONTROL_CLASS, className)} {...rest} />
}

export function Select({ className, ...rest }: SelectHTMLAttributes<HTMLSelectElement>) {
  return <select className={cn(CONTROL_CLASS, 'pr-1', className)} {...rest} />
}
