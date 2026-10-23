import type { Overview, ReasonCount, WaitSummary } from '@/api/types'
import { Card, CardHeader } from '@/components/ui/Card'
import { Stat } from '@/components/ui/Stat'
import { Table, Td, Th, Tr } from '@/components/ui/Table'
import { StatusBadge, VerdictCounts } from '@/components/ui/VerdictBadge'
import { EMPTY_VALUE, formatElapsed, formatInt, formatPercent, ratio } from '@/lib/format'
import { KIND_WORDS } from '@/lib/kinds'
import { explainReason } from '@/lib/labels'

function Wait({ summary }: { summary: WaitSummary | null }) {
  return summary?.p50 == null ? EMPTY_VALUE : formatElapsed(summary.p50)
}

function waitDetail(summary: WaitSummary | null, noun: string): string {
  if (summary === null) return `no ${noun} upload in the window`
  const within = summary.p90 === null ? '' : ` · 90% within ${formatElapsed(summary.p90)}`
  return `median of ${formatInt(summary.tasks)} ${noun}${within}`
}

const VERDICT_ORDER = { fail: 0, void: 1, pass: 2 } as const

function ReasonsTable({
  reasons,
  overview,
}: {
  reasons: readonly ReasonCount[]
  overview: Overview
}) {
  const total = reasons.reduce((sum, reason) => sum + reason.tasks, 0)
  const sorted = [...reasons].sort(
    (a, b) => VERDICT_ORDER[a.verdict] - VERDICT_ORDER[b.verdict] || b.tasks - a.tasks,
  )
  return (
    <Table>
      <thead>
        <tr>
          <Th>Result</Th>
          <Th>Reason</Th>
          <Th hint="Whether a validator checked the upload or it passed on the miner's own counts">
            Checked
          </Th>
          <Th align="right">Tasks</Th>
          <Th align="right" hint="Of every finalized upload of this kind">
            Share
          </Th>
        </tr>
      </thead>
      <tbody>
        {sorted.map((reason) => (
          <Tr key={`${reason.verdict}:${reason.reason}:${reason.checked}`}>
            <Td>
              <StatusBadge status={reason.verdict} />
            </Td>
            <Td className="min-w-64 whitespace-normal">
              <div className="font-mono text-xs">{reason.reason}</div>
              <div className="text-ink-muted text-xs">
                {explainReason(reason.reason, overview.kind) ??
                  (reason.reason === 'unchecked'
                    ? 'No validator checked it; it counted on the miner’s own report.'
                    : null)}
              </div>
            </Td>
            <Td className="text-ink-muted text-xs">{reason.checked ? 'Checked' : 'Unchecked'}</Td>
            <Td numeric>{formatInt(reason.tasks)}</Td>
            <Td numeric>{formatPercent(ratio(reason.tasks, total), 2)}</Td>
          </Tr>
        ))}
      </tbody>
    </Table>
  )
}

export function ChecksPanel({ overview }: { overview: Overview }) {
  const { window, checks, taken_back: takenBack, window_hours: hours, kind } = overview
  const finalized = window.pass + window.fail + window.void
  const words = KIND_WORDS[kind].credited
  return (
    <Card>
      <CardHeader
        title="Checks and pay"
        description={`How long ${kind} uploads waited for their verdict, how they ended and why, last ${hours} hours. A pass counts toward the miner’s share as soon as its verdict is final.`}
      />
      <div className="grid grid-cols-2 gap-3 p-4 md:grid-cols-4">
        <Stat
          label="Upload to verdict"
          value={<Wait summary={checks.verdict_s.unchecked} />}
          detail={waitDetail(checks.verdict_s.unchecked, 'unchecked uploads')}
        />
        <Stat
          label="Checked: upload to verdict"
          value={<Wait summary={checks.verdict_s.checked} />}
          detail={waitDetail(checks.verdict_s.checked, 'checked uploads')}
        />
        <Stat
          label={`Passed · ${hours} h`}
          value={formatPercent(ratio(window.pass, finalized))}
          detail={
            <>
              of {formatInt(finalized)} finalized ·{' '}
              <VerdictCounts pass={window.pass} fail={window.fail} void={window.void} />
            </>
          }
        />
        <Stat
          label={`Taken back · ${hours} h`}
          value={formatInt(takenBack.tasks)}
          detail={
            takenBack.tasks === 0
              ? 'no pass was taken back'
              : `passes from ${formatInt(takenBack.miners)} miners, ${formatInt(takenBack.credited)} ${words} uncounted`
          }
        />
      </div>
      {checks.reasons.length > 0 ? (
        <div className="border-line border-t">
          <ReasonsTable reasons={checks.reasons} overview={overview} />
        </div>
      ) : null}
    </Card>
  )
}
