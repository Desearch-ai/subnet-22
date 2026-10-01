import { pauseRequests, waitForPause } from './backoff'
import { API_BASE_URL, DEFAULT_RETRY_AFTER_S } from './config'

const HTTP_NOT_FOUND = 404
const HTTP_TOO_MANY_REQUESTS = 429
const HTTP_UNAVAILABLE = 503

export type QueryParams = Record<string, string | number | boolean | null | undefined>

export class ApiError extends Error {
  readonly status: number

  constructor(status: number, path: string) {
    super(`The task API answered ${status} for ${path}`)
    this.name = 'ApiError'
    this.status = status
  }

  get isNotFound(): boolean {
    return this.status === HTTP_NOT_FOUND
  }

  get isBackoff(): boolean {
    return this.status === HTTP_TOO_MANY_REQUESTS || this.status === HTTP_UNAVAILABLE
  }
}

export class UnreachableError extends Error {
  constructor() {
    super(`The task API at ${API_BASE_URL} could not be reached`)
    this.name = 'UnreachableError'
  }
}

export function buildUrl(path: string, params: QueryParams = {}): string {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value !== null && value !== undefined && value !== '') search.set(key, String(value))
  }
  const query = search.toString()
  return `${API_BASE_URL}${path}${query ? `?${query}` : ''}`
}

export function parseRetryAfter(header: string | null): number {
  const seconds = Number(header)
  return header !== null && Number.isFinite(seconds) && seconds > 0
    ? seconds
    : DEFAULT_RETRY_AFTER_S
}

export async function apiGet<T>(
  path: string,
  params?: QueryParams,
  signal?: AbortSignal,
): Promise<T> {
  await waitForPause(signal)
  let response: Response
  try {
    response = await fetch(buildUrl(path, params), {
      headers: { Accept: 'application/json' },
      ...(signal ? { signal } : {}),
    })
  } catch (error) {
    if (signal?.aborted) throw error
    throw new UnreachableError()
  }
  if (!response.ok) {
    const error = new ApiError(response.status, path)
    if (error.isBackoff) pauseRequests(parseRetryAfter(response.headers.get('Retry-After')))
    throw error
  }
  return (await response.json()) as T
}
