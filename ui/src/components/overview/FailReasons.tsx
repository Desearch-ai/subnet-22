import type { Overview } from '@/api/types'
import { Card, CardHeader } from '@/components/ui/Card'
import { EmptyState } from '@/components/ui/States'
import { Table, Td, Th, Tr } from '@/components/ui/Table'
import { StatusBadge } from '@/components/ui/VerdictBadge'
import { formatInt } from '@/lib/format'
import { explainReason } from '@/lib/labels'

export function FailReasons({ overview }: { overview: Overview }) {
  const { checks, window_hours: hours, kind } = overview
  const notPassed = checks.reasons
    .filter((reason) => reason.verdict !== 'pass')
    .sort((a, b) => b.tasks - a.tasks)
  return (
    <Card>
      <CardHeader
        title="Why uploads did not pass"
        description={`${kind === 'embed' ? 'Embed' : 'Crawl'} uploads that failed or ended void, last ${hours} hours`}
      />
      {notPassed.length === 0 ? (
        <EmptyState title="Every upload passed." />
      ) : (
        <Table>
          <thead>
            <tr>
              <Th>Result</Th>
              <Th>Why</Th>
              <Th align="right">Uploads</Th>
            </tr>
          </thead>
          <tbody>
            {notPassed.map((reason) => (
              <Tr key={`${reason.verdict}:${reason.reason}:${reason.checked}`}>
                <Td>
                  <StatusBadge status={reason.verdict} />
                </Td>
                <Td className="whitespace-normal" title={reason.reason}>
                  {explainReason(reason.reason, kind) ?? reason.reason}
                </Td>
                <Td numeric>{formatInt(reason.tasks)}</Td>
              </Tr>
            ))}
          </tbody>
        </Table>
      )}
    </Card>
  )
}
