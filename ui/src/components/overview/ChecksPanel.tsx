import type { Overview } from '@/api/types'
import { Card, CardHeader } from '@/components/ui/Card'
import { Stat } from '@/components/ui/Stat'
import { EmptyState } from '@/components/ui/States'
import { Table, Td, Th, Tr } from '@/components/ui/Table'
import { StatusBadge } from '@/components/ui/VerdictBadge'
import { formatInt, formatPercent, ratio } from '@/lib/format'
import { explainReason } from '@/lib/labels'

export function ChecksPanel({ overview }: { overview: Overview }) {
  const { window, checks, taken_back: takenBack, window_hours: hours, kind } = overview
  const finalized = window.pass + window.fail + window.void
  const notPassed = checks.reasons
    .filter((reason) => reason.verdict !== 'pass')
    .sort((a, b) => b.tasks - a.tasks)
  return (
    <Card>
      <CardHeader
        title="Verdicts"
        description={`${kind === 'embed' ? 'Embed' : 'Crawl'} uploads, last ${hours} hours. What passes counts toward the miner’s share as soon as its verdict is in.`}
      />
      <div className="grid gap-3 p-4 sm:grid-cols-2">
        <Stat
          label="Passed"
          value={formatPercent(ratio(window.pass, finalized), 2)}
          detail={`${formatInt(window.pass)} of ${formatInt(finalized)} uploads`}
        />
        <Stat
          label="Taken back"
          value={formatInt(takenBack.tasks)}
          detail="passes taken back after a later check failed"
        />
      </div>
      <div className="border-line border-t">
        <h3 className="text-ink px-4 pt-3 text-sm font-medium">Why uploads did not pass</h3>
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
      </div>
    </Card>
  )
}
