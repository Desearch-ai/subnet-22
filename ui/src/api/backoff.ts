type Listener = () => void

let pausedUntil = 0
const listeners = new Set<Listener>()

export function pauseRequests(seconds: number): void {
  pausedUntil = Math.max(pausedUntil, Date.now() + seconds * 1000)
  listeners.forEach((listener) => {
    listener()
  })
}

export function getPausedUntil(): number {
  return pausedUntil
}

export function subscribeToPause(listener: Listener): () => void {
  listeners.add(listener)
  return () => {
    listeners.delete(listener)
  }
}

export async function waitForPause(signal?: AbortSignal): Promise<void> {
  while (Date.now() < pausedUntil) {
    const delay = pausedUntil - Date.now()
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(resolve, delay)
      signal?.addEventListener(
        'abort',
        () => {
          clearTimeout(timer)
          reject(signal.reason instanceof Error ? signal.reason : new Error('aborted'))
        },
        { once: true },
      )
    })
  }
}
