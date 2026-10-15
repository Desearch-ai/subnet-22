import type { Kind, TaskScore } from '@/api/types'
import { Badge } from '@/components/ui/Badge'
import { Identifier } from '@/components/ui/Identifier'
import { Table, Td, Th, Tr } from '@/components/ui/Table'
import { TimeAgo } from '@/components/ui/TimeAgo'
import { StatusBadge } from '@/components/ui/VerdictBadge'
import { formatInt } from '@/lib/format'
import { countColumns } from '@/lib/kinds'
import { explainReason } from '@/lib/labels'
import { minerPath } from '@/lib/paths'

interface UploadsTableProps {
  uploads: readonly TaskScore[]
  kind: Kind
}

export function UploadsTable({ uploads, kind }: UploadsTableProps) {
  const { credited } = countColumns(kind)
  return (
    <Table>
      <thead>
        <tr>
          <Th>Finalized</Th>
          <Th>Miner</Th>
          <Th>Result</Th>
          <Th>Reason</Th>
          <Th align="right" hint={credited.hint}>
            {credited.label}
          </Th>
          <Th />
        </tr>
      </thead>
      <tbody>
        {uploads.map((upload, index) => (
          <Tr key={`${upload.miner}:${upload.scored_at}`}>
            <Td className="text-ink-muted text-xs">
              <TimeAgo at={upload.scored_at} />
            </Td>
            <Td>
              <Identifier
                value={upload.miner}
                label="miner hotkey"
                uid={upload.miner_uid}
                to={minerPath(upload.miner)}
              />
            </Td>
            <Td>
              <StatusBadge status={upload.verdict} />
            </Td>
            <Td
              className="text-ink-muted font-mono text-xs"
              title={explainReason(upload.reason, upload.kind) ?? undefined}
            >
              {upload.reason}
            </Td>
            <Td numeric>{formatInt(upload.credited)}</Td>
            <Td>{index === 0 ? <Badge tone="accent">Shown on this page</Badge> : null}</Td>
          </Tr>
        ))}
      </tbody>
    </Table>
  )
}
