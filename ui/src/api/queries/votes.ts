import { useInfiniteQuery } from '@tanstack/react-query'
import { apiGet } from '../client'
import { PAGE_SIZE } from '../config'
import type { VotesResponse } from '../types'
import { FIRST_PAGE, nextPageParam, pollFirstPageOnly } from './paging'

export interface VoteFilters {
  validator: string
  agreed?: boolean
}

export function useVotes(filters: VoteFilters) {
  return useInfiniteQuery({
    queryKey: ['votes', filters],
    queryFn: ({ pageParam, signal }) =>
      apiGet<VotesResponse>(
        '/v1/votes',
        { ...filters, limit: PAGE_SIZE, before: pageParam },
        signal,
      ),
    initialPageParam: FIRST_PAGE,
    getNextPageParam: nextPageParam,
    refetchInterval: (query) => pollFirstPageOnly(query.state.data),
    placeholderData: (previous, previousQuery) =>
      (previousQuery?.queryKey[1] as VoteFilters | undefined)?.validator === filters.validator
        ? previous
        : undefined,
  })
}
