import { Link, useRouteError } from 'react-router'
import { OVERVIEW_PATH } from '@/lib/paths'

export function RouteError() {
  const error = useRouteError()
  return (
    <div role="alert" className="mx-auto max-w-xl px-4 py-16 text-sm">
      <h1 className="text-ink text-xl font-medium">This page could not be shown</h1>
      <p className="text-ink-muted mt-2">
        {error instanceof Error ? error.message : 'An unexpected error happened.'}
      </p>
      <Link to={OVERVIEW_PATH} className="text-accent mt-4 inline-block hover:underline">
        Back to the overview
      </Link>
    </div>
  )
}
