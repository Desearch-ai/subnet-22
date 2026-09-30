import { useEffect, useState } from 'react'

export const TICK_SECOND_MS = 1_000
export const TICK_RELATIVE_MS = 15_000

export function useNow(intervalMs: number = TICK_RELATIVE_MS): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const timer = setInterval(() => {
      setNow(Date.now())
    }, intervalMs)
    return () => {
      clearInterval(timer)
    }
  }, [intervalMs])
  return now
}
