import type { Kind, Verdict } from '@/api/types'
import { DEFAULT_KIND, KIND_PARAM } from './kinds'

export const OVERVIEW_PATH = '/'
export const TASKS_PATH = '/tasks'
export const MINERS_PATH = '/miners'
export const VALIDATORS_PATH = '/validators'

export function taskPath(taskId: string): string {
  return `${TASKS_PATH}/${encodeURIComponent(taskId)}`
}

export function minerPath(hotkey: string): string {
  return `${MINERS_PATH}/${encodeURIComponent(hotkey)}`
}

export function validatorPath(hotkey: string): string {
  return `${VALIDATORS_PATH}/${encodeURIComponent(hotkey)}`
}

export function withKind(path: string, kind: Kind): string {
  return kind === DEFAULT_KIND ? path : `${path}?${KIND_PARAM}=${kind}`
}

export function tasksPath(verdict: Verdict, kind: Kind): string {
  return `${TASKS_PATH}?verdict=${verdict}&${KIND_PARAM}=${kind}`
}
