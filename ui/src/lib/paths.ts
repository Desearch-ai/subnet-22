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
