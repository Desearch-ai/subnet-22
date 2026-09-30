import { useInfiniteQuery, useQuery } from '@tanstack/react-query'
import { apiGet } from '../client'
import { PAGE_SIZE, POLL_INTERVAL_MS } from '../config'
import type { TaskDetail, TasksResponse, Verdict } from '../types'
import { FIRST_PAGE, nextPageParam, pollFirstPageOnly } from './paging'

export interface TaskFilters {
  miner?: string
  validator?: string
  verdict?: Verdict
}

const FINAL_STATUSES: ReadonlySet<string> = new Set<Verdict>(['pass', 'fail', 'void'])

export function fetchTask(taskId: string, signal?: AbortSignal): Promise<TaskDetail> {
  return apiGet<TaskDetail>(`/v1/tasks/${encodeURIComponent(taskId)}`, {}, signal)
}

export function useTasks(filters: TaskFilters, limit = PAGE_SIZE) {
  return useInfiniteQuery({
    queryKey: ['tasks', filters, limit],
    queryFn: ({ pageParam, signal }) =>
      apiGet<TasksResponse>('/v1/tasks', { ...filters, limit, before: pageParam }, signal),
    initialPageParam: FIRST_PAGE,
    getNextPageParam: nextPageParam,
    refetchInterval: (query) => pollFirstPageOnly(query.state.data),
  })
}

export function useTask(taskId: string) {
  return useQuery({
    queryKey: ['task', taskId],
    queryFn: ({ signal }) => fetchTask(taskId, signal),
    refetchInterval: (query) =>
      query.state.data && FINAL_STATUSES.has(query.state.data.status) ? false : POLL_INTERVAL_MS,
  })
}
