import { HOURS_SHOWN, WINDOW_HOURS } from '../config'
import type { Kind, Verdict } from '../types'

export interface TaskFilters {
  miner?: string
  validator?: string
  verdict?: Verdict
  kind?: Kind
}

export interface VoteFilters {
  validator: string
  agreed?: boolean
}

export interface HoursFilters {
  kind: Kind
  miner?: string
}

export const queryKeys = {
  overview: (kind: Kind) => ['overview', WINDOW_HOURS, kind] as const,
  live: () => ['live'] as const,
  hours: ({ kind, miner }: HoursFilters) => ['hours', HOURS_SHOWN, kind, miner ?? null] as const,
  miners: (kind: Kind) => ['miners', WINDOW_HOURS, kind] as const,
  miner: (hotkey: string) => ['miner', hotkey] as const,
  tasks: (filters: TaskFilters, limit: number) => ['tasks', filters, limit] as const,
  task: (taskId: string) => ['task', taskId] as const,
  validators: () => ['validators', WINDOW_HOURS] as const,
  validator: (hotkey: string) => ['validator', hotkey, WINDOW_HOURS] as const,
  votes: (filters: VoteFilters) => ['votes', filters] as const,
}

/** A list keeps showing while only its result filter changes; other filters change what its columns mean. */
export function keepsTaskList(before: TaskFilters | undefined, after: TaskFilters): boolean {
  return (
    before !== undefined &&
    before.miner === after.miner &&
    before.validator === after.validator &&
    before.kind === after.kind
  )
}
