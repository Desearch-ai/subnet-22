import type { InfiniteData, UseInfiniteQueryResult } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import { Button } from './Button'
import { EmptyState, ErrorState, LoadingState, StaleNotice } from './States'

interface PagedListProps<Page, Item> {
  query: UseInfiniteQueryResult<InfiniteData<Page>>
  itemsOf: (page: Page) => Item[]
  emptyTitle: string
  emptyDetail?: string
  children: (items: Item[]) => ReactNode
}

export function PagedList<Page, Item>({
  query,
  itemsOf,
  emptyTitle,
  emptyDetail,
  children,
}: PagedListProps<Page, Item>) {
  if (query.data === undefined) {
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
  const items = query.data.pages.flatMap(itemsOf)
  if (items.length === 0) return <EmptyState title={emptyTitle}>{emptyDetail}</EmptyState>
  return (
    <>
      {query.isError ? <StaleNotice /> : null}
      {children(items)}
      {query.hasNextPage ? (
        <div className="border-line flex items-center gap-3 border-t px-4 py-3">
          <Button
            disabled={query.isFetchingNextPage}
            onClick={() => {
              void query.fetchNextPage()
            }}
          >
            {query.isFetchingNextPage ? 'Loading…' : 'Load more'}
          </Button>
          <span className="text-ink-faint text-xs">{items.length} shown</span>
        </div>
      ) : null}
    </>
  )
}
