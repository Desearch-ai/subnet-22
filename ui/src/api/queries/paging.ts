import type { InfiniteData } from '@tanstack/react-query'
import { POLL_INTERVAL_MS } from '../config'

interface Page {
  next: number | null
}

export const FIRST_PAGE: number | null = null

export function nextPageParam(page: Page): number | undefined {
  return page.next ?? undefined
}

export function pollFirstPageOnly<P>(data: InfiniteData<P> | undefined): number | false {
  return data !== undefined && data.pages.length > 1 ? false : POLL_INTERVAL_MS
}
