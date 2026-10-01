const DEFAULT_API_BASE_URL = 'http://127.0.0.1:18080'

const configured: unknown = import.meta.env.VITE_TASK_API_URL

export const API_BASE_URL = (
  typeof configured === 'string' && configured !== '' ? configured : DEFAULT_API_BASE_URL
).replace(/\/+$/, '')

export const WINDOW_HOURS = 24
export const POLL_INTERVAL_MS = 60_000
export const PAGE_SIZE = 50
export const DEFAULT_RETRY_AFTER_S = 10
export const MAX_RETRIES = 3
