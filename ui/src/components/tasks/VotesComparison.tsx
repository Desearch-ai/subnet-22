import type { ReactNode } from 'react'
import type { Kind, Vote } from '@/api/types'
import { Badge } from '@/components/ui/Badge'
import { Neuron } from '@/components/ui/Neuron'
import { EmptyState } from '@/components/ui/States'
import { Table } from '@/components/ui/Table'
import { TimeAgo } from '@/components/ui/TimeAgo'
import { StatusBadge } from '@/components/ui/VerdictBadge'
import { cn } from '@/lib/cn'
import { formatInt } from '@/lib/format'
import { validatorPath } from '@/lib/paths'

type Comparable = string | number

interface Metric {
  label: string
  hint?: string
  valueOf: (vote: Vote) => Comparable
  render?: (vote: Vote) => ReactNode
}

const SHARED_METRICS: readonly Metric[] = [
  {
    label: 'Result',
    valueOf: (vote) => vote.verdict,
    render: (vote) => <StatusBadge status={vote.verdict} />,
  },
  { label: 'Reason', valueOf: (vote) => vote.reason },
]

const KIND_METRICS: Record<Kind, readonly Metric[]> = {
  crawl: [
    ...SHARED_METRICS,
    { label: 'Rows returned', valueOf: (vote) => vote.returned },
    {
      label: 'Checked pages',
      hint: 'Pages this validator fetched again',
      valueOf: (vote) => vote.sampled,
    },
    { label: 'Matched', valueOf: (vote) => vote.matched },
    { label: 'Mismatched', valueOf: (vote) => vote.mismatched },
    { label: 'Unverifiable', valueOf: (vote) => vote.unverifiable },
    { label: 'Errors confirmed', valueOf: (vote) => vote.errors_confirmed },
    { label: 'Errors not confirmed', valueOf: (vote) => vote.errors_unconfirmed },
    {
      label: 'Rows counted',
      hint: 'Rows this validator would pay for',
      valueOf: (vote) => vote.credited,
    },
  ],
  embed: [
    ...SHARED_METRICS,
    { label: 'Texts returned', valueOf: (vote) => vote.returned },
    {
      label: 'Checked texts',
      hint: 'Texts this validator embedded again',
      valueOf: (vote) => vote.sampled,
    },
    {
      label: 'Matched',
      hint: 'Checked texts that reached a similarity of 0.999',
      valueOf: (vote) => vote.matched,
    },
    { label: 'Mismatched', valueOf: (vote) => vote.mismatched },
    {
      label: 'Characters counted',
      hint: 'Characters this validator would pay for',
      valueOf: (vote) => vote.credited,
    },
  ],
}

function VoteHeading({ vote }: { vote: Vote }) {
  return (
    <th
      scope="col"
      className={cn(
        'border-line min-w-44 border-b border-l px-3 py-2 text-left align-top font-normal',
        vote.decided ? 'bg-accent-wash' : null,
      )}
    >
      <Neuron
        hotkey={vote.validator}
        label="validator hotkey"
        uid={vote.validator_uid}
        coldkey={vote.validator_coldkey}
        to={validatorPath(vote.validator)}
      />
      <div className="mt-1.5 flex flex-wrap gap-1">
        {vote.decided ? (
          <Badge tone="accent" title="The final result was taken from this vote">
            Decided
          </Badge>
        ) : null}
        {vote.agreed === true && !vote.decided ? <Badge tone="pass">Agreed</Badge> : null}
        {vote.agreed === false ? (
          <Badge tone="fail" title="This vote did not match the final result">
            Disagreed
          </Badge>
        ) : null}
        {vote.agreed === null ? (
          <Badge title="No final result to compare this vote with">Not compared</Badge>
        ) : null}
      </div>
      <div className="text-ink-faint mt-1.5 text-xs">
        voted <TimeAgo at={vote.voted_at} />
      </div>
    </th>
  )
}

function summarize(votes: readonly Vote[]): string {
  const voters = `${votes.length} ${votes.length === 1 ? 'validator' : 'validators'} voted.`
  if (votes.every((vote) => vote.agreed === null)) {
    return `${voters} No majority formed, so the votes are not compared with a result.`
  }
  const disagreed = votes.filter((vote) => vote.agreed === false).length
  const outcome =
    disagreed === 0
      ? 'None disagreed with the final result.'
      : `${disagreed} disagreed with the final result.`
  return `${voters} ${outcome} Values that differ from the deciding vote are marked.`
}

interface VotesComparisonProps {
  votes: readonly Vote[]
  kind: Kind
}

export function VotesComparison({ votes, kind }: VotesComparisonProps) {
  if (votes.length === 0) {
    return <EmptyState title="No votes are recorded for this upload." />
  }
  const deciding = votes.find((vote) => vote.decided)
  return (
    <>
      <p className="text-ink-muted border-line border-b px-4 py-2 text-sm">{summarize(votes)}</p>
      <Table>
        <thead>
          <tr>
            <th scope="col" className="border-line relative border-b px-3 py-2">
              <span className="sr-only">Measure</span>
            </th>
            {votes.map((vote) => (
              <VoteHeading key={vote.validator} vote={vote} />
            ))}
          </tr>
        </thead>
        <tbody>
          {KIND_METRICS[kind].map((metric) => (
            <tr key={metric.label} className="[&:last-child>*]:border-b-0">
              <th
                scope="row"
                title={metric.hint}
                className="border-line text-ink-muted border-b px-3 py-1.5 text-left text-sm font-normal whitespace-nowrap"
              >
                {metric.label}
              </th>
              {votes.map((vote) => {
                const value = metric.valueOf(vote)
                const differs = deciding !== undefined && metric.valueOf(deciding) !== value
                return (
                  <td
                    key={vote.validator}
                    className={cn(
                      'border-line relative border-b border-l px-3 py-1.5 font-mono text-xs whitespace-nowrap tabular-nums',
                      differs ? 'bg-fail-wash' : null,
                    )}
                  >
                    {metric.render?.(vote) ??
                      (typeof value === 'number' ? formatInt(value) : value)}
                    {differs ? (
                      <span className="sr-only"> (differs from the deciding vote)</span>
                    ) : null}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </Table>
    </>
  )
}
