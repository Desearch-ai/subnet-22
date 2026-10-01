import type { InfiniteData, UseInfiniteQueryResult } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import { cn } from '@/lib/cn'
import { Button } from './Button'
import { EmptyState, ErrorState, LoadingState, Spinner, StaleNotice } from './States'

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
      <div
        aria-busy={query.isPlaceholderData}
        className={cn('transition-opacity', query.isPlaceholderData ? 'opacity-50' : null)}
      >
        {children(items)}
      </div>
      {query.hasNextPage ? (
        <div className="border-line flex items-center gap-3 border-t px-4 py-3">
          <Button
            disabled={query.isFetchingNextPage}
            onClick={() => {
              void query.fetchNextPage()
            }}
          >
            {query.isFetchingNextPage ? <Spinner className="size-3.5" /> : null}
            {query.isFetchingNextPage ? 'Loading…' : 'Load more'}
          </Button>
          <span className="text-ink-faint text-xs">{items.length} shown</span>
        </div>
      ) : null}
    </>
  )
}
