import { useLive } from '@/api/queries/overview'
import type { LiveClaim, LiveUpload } from '@/api/types'
import { Card, CardHeader } from '@/components/ui/Card'
import { Identifier } from '@/components/ui/Identifier'
import { KindBadge } from '@/components/ui/KindBadge'
import { QueryState } from '@/components/ui/QueryState'
import { EmptyState } from '@/components/ui/States'
import { Table, Td, Th, Tr } from '@/components/ui/Table'
import { TimeAgo } from '@/components/ui/TimeAgo'
import { TICK_SECOND_MS, useNow } from '@/hooks/useNow'
import { formatAbsolute, formatCountdown, formatInt } from '@/lib/format'
import { formatTaskSize } from '@/lib/kinds'
import { minerPath, taskPath, validatorPath } from '@/lib/paths'

const SIZE_HINT = 'URLs in a crawl task, texts in an embed task'

function Countdown({ until }: { until: number }) {
  const now = useNow(TICK_SECOND_MS)
  return (
    <span title={formatAbsolute(until)} className="cursor-help font-mono tabular-nums">
      {formatCountdown(until, now)}
    </span>
  )
}

function ClaimsTable({ claims }: { claims: readonly LiveClaim[] }) {
  if (claims.length === 0) return <EmptyState title="No miner holds a task right now." />
  return (
    <Table>
      <thead>
        <tr>
          <Th>Task</Th>
          <Th>Kind</Th>
          <Th>Miner</Th>
          <Th align="right" hint={SIZE_HINT}>
            Size
          </Th>
          <Th align="right" hint="Time left to upload before the task goes back to the queue">
            Time left
          </Th>
        </tr>
      </thead>
      <tbody>
        {claims.map((claim) => (
          <Tr key={claim.task_id}>
            <Td>
              <Identifier value={claim.task_id} label="task id" to={taskPath(claim.task_id)} />
            </Td>
            <Td>
              <KindBadge kind={claim.kind} />
            </Td>
            <Td>
              <Identifier
                value={claim.miner}
                label="miner hotkey"
                uid={claim.miner_uid}
                to={minerPath(claim.miner)}
              />
            </Td>
            <Td numeric>{formatTaskSize(claim.kind, claim.urls, claim.texts)}</Td>
            <Td align="right">
              <Countdown until={claim.expires_at} />
            </Td>
          </Tr>
        ))}
      </tbody>
    </Table>
  )
}

function UploadsTable({ uploads }: { uploads: readonly LiveUpload[] }) {
  if (uploads.length === 0) return <EmptyState title="No upload is waiting for votes." />
  return (
    <Table>
      <thead>
        <tr>
          <Th>Task</Th>
          <Th>Kind</Th>
          <Th>Miner</Th>
          <Th align="right" hint={SIZE_HINT}>
            Size
          </Th>
          <Th>Uploaded</Th>
          <Th hint="Validators that have voted, out of those that check this kind of task">
            Voted
          </Th>
          <Th align="right" hint="Time left for validators to vote">
            Time left
          </Th>
        </tr>
      </thead>
      <tbody>
        {uploads.map((upload) => (
          <Tr key={upload.task_id}>
            <Td>
              <Identifier value={upload.task_id} label="task id" to={taskPath(upload.task_id)} />
            </Td>
            <Td>
              <KindBadge kind={upload.kind} />
            </Td>
            <Td>
              <Identifier
                value={upload.miner}
                label="miner hotkey"
                uid={upload.miner_uid}
                to={minerPath(upload.miner)}
              />
            </Td>
            <Td numeric>{formatTaskSize(upload.kind, upload.urls, upload.texts)}</Td>
            <Td className="text-ink-muted text-xs">
              <TimeAgo at={upload.completed_at} />
            </Td>
            <Td>
              <span className="flex flex-col gap-0.5">
                <span className="text-ink-muted text-xs">
                  {upload.voters.length} of {upload.electorate} voted
                </span>
                {upload.voters.map((voter) => (
                  <Identifier
                    key={voter.hotkey}
                    value={voter.hotkey}
                    label="validator hotkey"
                    uid={voter.uid}
                    to={validatorPath(voter.hotkey)}
                  />
                ))}
              </span>
            </Td>
            <Td align="right">
              <Countdown until={upload.deadline} />
            </Td>
          </Tr>
        ))}
      </tbody>
    </Table>
  )
}

function shown(listed: number | undefined, total: number | undefined): string {
  return listed !== undefined && total !== undefined && total > listed
    ? `, the oldest ${formatInt(listed)} of ${formatInt(total)}`
    : ''
}

export function LivePanel() {
  const query = useLive()
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card>
        <CardHeader
          title="Claimed now"
          description={`Tasks of both kinds a miner holds and has not uploaded yet${shown(query.data?.claims.length, query.data?.claims_total)}`}
        />
        <QueryState query={query}>{(live) => <ClaimsTable claims={live.claims} />}</QueryState>
      </Card>
      <Card>
        <CardHeader
          title="Being checked now"
          description={`Uploads of both kinds validators are voting on${shown(query.data?.uploads.length, query.data?.uploads_total)}`}
        />
        <QueryState query={query}>{(live) => <UploadsTable uploads={live.uploads} />}</QueryState>
      </Card>
    </div>
  )
}
