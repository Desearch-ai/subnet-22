import { Link } from 'react-router'
import { Card } from '@/components/ui/Card'
import { EmptyState } from '@/components/ui/States'
import { OVERVIEW_PATH } from '@/lib/paths'

export function NotFoundPage() {
  return (
    <Card>
      <EmptyState title="There is no page at this address.">
        <Link to={OVERVIEW_PATH} className="text-accent hover:underline">
          Back to the overview
        </Link>
      </EmptyState>
    </Card>
  )
}
