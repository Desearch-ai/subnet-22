import type { ReactNode } from 'react'
import type { TaskDetail, TaskScore, TaskStatus, Verdict } from '@/api/types'
import { Badge, type Tone } from '@/components/ui/Badge'
import { Card } from '@/components/ui/Card'
import { Identifier } from '@/components/ui/Identifier'
import { KindBadge } from '@/components/ui/KindBadge'
import { Neuron } from '@/components/ui/Neuron'
import { TimeAgo } from '@/components/ui/TimeAgo'
import { EMPTY_VALUE, formatElapsed } from '@/lib/format'
import { explainReason, STATUS_MEANING, VERDICT_MEANING } from '@/lib/labels'
import { minerPath, validatorPath } from '@/lib/paths'

type OpenStatus = Exclude<TaskStatus, Verdict>

const OPEN_TITLE: Record<OpenStatus, string> = {
  queued: 'Queued',
  claimed: 'Claimed',
  open: 'Uploaded, waiting for votes',
  voting: 'Being checked',
}

const FINAL_TITLE: Record<Verdict, string> = {
  pass: 'Passed',
  fail: 'Failed',
  void: 'Ended void',
}

const STATUS_TONE: Record<TaskStatus, Tone> = {
  queued: 'neutral',
  claimed: 'accent',
  open: 'accent',
  voting: 'accent',
  pass: 'pass',
  fail: 'fail',
  void: 'void',
}

function isFinal(status: TaskStatus): status is Verdict {
  return status === 'pass' || status === 'fail' || status === 'void'
}

function why(score: TaskScore): string {
  return explainReason(score.reason, score.kind) ?? VERDICT_MEANING[score.verdict]
}

function Block({ title, hint, children }: { title: string; hint?: string; children: ReactNode }) {
  return (
    <div className="min-w-0">
      <div
        className="text-ink-faint text-label mb-1 font-mono tracking-wider uppercase"
        title={hint}
      >
        {title}
      </div>
      {children}
    </div>
  )
}

function Step({ label, children }: { label: string; children: ReactNode }) {
  return (
    <li className="flex items-baseline gap-2">
      <span
        aria-hidden
        className="bg-line-strong size-1.5 shrink-0 translate-y-[-1px] rounded-full"
      />
      <span className="text-ink-muted w-20 shrink-0 text-xs">{label}</span>
      <span className="text-ink text-xs">{children}</span>
    </li>
  )
}

function Timeline({ score }: { score: TaskScore }) {
  const { claimed_at: claimed, completed_at: uploaded, scored_at: decided } = score
  return (
    <ol className="space-y-1">
      <Step label="Claimed">{claimed === null ? EMPTY_VALUE : <TimeAgo at={claimed} />}</Step>
      <Step label="Uploaded">
        {uploaded === null ? (
          EMPTY_VALUE
        ) : claimed === null ? (
          <TimeAgo at={uploaded} />
        ) : (
          `${formatElapsed(uploaded - claimed)} after the claim`
        )}
      </Step>
      <Step label="Verdict">
        {uploaded === null ? (
          <TimeAgo at={decided} />
        ) : (
          `${formatElapsed(decided - uploaded)} after the upload`
        )}
      </Step>
    </ol>
  )
}

export function TaskHeader({ task }: { task: TaskDetail }) {
  const { status, score } = task
  const final = isFinal(status)
  const miner = final && score ? score.miner : task.miner
  const minerUid = final && score ? score.miner_uid : task.miner_uid
  const minerColdkey = final && score ? score.miner_coldkey : task.miner_coldkey
  return (
    <Card>
      <div className="border-line border-b px-4 py-3">
        <div className="flex flex-wrap items-center gap-2">
          <Badge tone={STATUS_TONE[status]}>
            {final ? FINAL_TITLE[status] : OPEN_TITLE[status]}
          </Badge>
          <KindBadge kind={task.kind} />
          <Identifier value={task.task_id} label="task id" />
        </div>
        <p className="text-ink mt-2 text-base">
          {final && score ? why(score) : STATUS_MEANING[status]}
        </p>
        {!final && score ? (
          <p className="text-ink-muted mt-1 text-sm">
            Its last upload {FINAL_TITLE[score.verdict].toLowerCase()}: {why(score)} The details
            below are from that upload.
          </p>
        ) : null}
      </div>
      <div className="grid gap-4 p-4 md:grid-cols-3">
        <Block
          title="Miner"
          hint={final ? 'The miner whose upload this is' : 'The miner holding the task now'}
        >
          {miner === null || status === 'queued' ? (
            <span className="text-ink-faint text-xs">no miner holds it now</span>
          ) : (
            <Neuron
              hotkey={miner}
              label="miner hotkey"
              uid={minerUid}
              coldkey={minerColdkey}
              to={minerPath(miner)}
            />
          )}
        </Block>
        <Block title="Decided by" hint="The validator whose report became the final result">
          {score ? (
            <Neuron
              hotkey={score.validator}
              label="validator hotkey"
              uid={score.validator_uid}
              coldkey={score.validator_coldkey}
              to={validatorPath(score.validator)}
            />
          ) : (
            <span className="text-ink-faint text-xs">no verdict yet</span>
          )}
        </Block>
        <Block title={final ? 'Timeline' : 'Last upload timeline'}>
          {score ? (
            <Timeline score={score} />
          ) : (
            <span className="text-ink-faint text-xs">{EMPTY_VALUE}</span>
          )}
        </Block>
      </div>
      <div className="border-line text-ink-faint border-t px-4 py-2 font-mono text-xs">
        round {task.round_id ?? EMPTY_VALUE} · {task.uploads.length}{' '}
        {task.uploads.length === 1 ? 'upload' : 'uploads'} finalized
      </div>
    </Card>
  )
}
