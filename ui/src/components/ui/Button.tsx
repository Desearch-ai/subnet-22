import type { ButtonHTMLAttributes } from 'react'
import { cn } from '@/lib/cn'

type Variant = 'primary' | 'outline'

const VARIANT_CLASS: Record<Variant, string> = {
  primary: 'bg-accent-solid text-accent-ink hover:bg-accent-solid-hover',
  outline: 'border border-line-strong text-ink hover:bg-raised',
}

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant
}

export function Button({ variant = 'outline', className, type = 'button', ...rest }: ButtonProps) {
  return (
    <button
      type={type}
      className={cn(
        'inline-flex h-8 items-center gap-2 rounded-md px-3 font-mono text-xs tracking-wide whitespace-nowrap uppercase transition-colors disabled:opacity-50',
        VARIANT_CLASS[variant],
        className,
      )}
      {...rest}
    />
  )
}
