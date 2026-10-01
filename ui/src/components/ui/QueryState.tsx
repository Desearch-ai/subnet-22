import type { UseQueryResult } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import { ErrorState, LoadingState, StaleNotice } from './States'

interface QueryStateProps<Data> {
  query: UseQueryResult<Data>
  children: (data: Data) => ReactNode
}

export function QueryState<Data>({ query, children }: QueryStateProps<Data>) {
  if (query.data !== undefined) {
    return (
      <>
        {query.isError ? <StaleNotice /> : null}
        {children(query.data)}
      </>
    )
  }
  if (query.isError) {
    return (
      <ErrorState
        error={query.error}
        onRetry={() => {
          void query.refetch()
        }}
      />
    )
  }
  return <LoadingState />
}
