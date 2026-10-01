import type { TaskUrl } from '@/api/types'
import { splitAtDifference } from '@/lib/diff'
import { EMPTY_VALUE, formatInt, formatScore } from '@/lib/format'

interface FactProps {
  label: string
  value: string
  hint?: string
}

function Fact({ label, value, hint }: FactProps) {
  return (
    <div title={hint} className={hint ? 'cursor-help' : undefined}>
      <dt className="text-ink-faint text-label font-mono tracking-wider uppercase">{label}</dt>
      <dd className="text-ink font-mono text-xs tabular-nums">{value}</dd>
    </div>
  )
}

interface TextPaneProps {
  title: string
  text: string | null
  compareWith?: string | null
}

function TextPane({ title, text, compareWith }: TextPaneProps) {
  const split = text !== null && compareWith ? splitAtDifference(text, compareWith) : null
  return (
    <div className="min-w-0">
      <div className="text-ink-faint text-label mb-1 font-mono tracking-wider uppercase">
        {title}
      </div>
      <pre className="border-line bg-ground scroll-thin max-h-48 overflow-auto rounded-md border px-2.5 py-2 font-mono text-xs break-words whitespace-pre-wrap">
        {text === null ? (
          <span className="text-ink-faint">nothing recorded</span>
        ) : split ? (
          <>
            <span className="text-ink-muted">{split.same}</span>
            <mark className="bg-fail-wash text-ink">{split.different}</mark>
          </>
        ) : (
          text
        )}
      </pre>
    </div>
  )
}

export function UrlDetail({ row }: { row: TaskUrl }) {
  const hasSnippets = row.miner_snippet !== null || row.validator_snippet !== null
  const hasWindows = row.miner_window !== null || row.validator_window !== null
  return (
    <div className="sticky left-0 max-w-[min(calc(100vw-2.25rem),86rem)] space-y-3 px-3 py-3 whitespace-normal">
      <p className="text-sm">
        {row.why ??
          (row.sampled
            ? 'The validator recorded no note for this page.'
            : 'This page was not part of the checked sample, so there is nothing to compare.')}
      </p>
      {row.sampled ? (
        <dl className="grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-4 lg:grid-cols-8">
          <Fact
            label="Similarity"
            value={formatScore(row.similarity)}
            hint="How close the two texts are, from 0 to 1"
          />
          <Fact
            label="Precision"
            value={formatScore(row.precision)}
            hint="Share of the miner's text that is in the validator's text"
          />
          <Fact
            label="Recall"
            value={formatScore(row.recall)}
            hint="Share of the validator's text that is in the miner's text"
          />
          <Fact
            label="Growth"
            value={formatScore(row.growth)}
            hint="Length of the miner's text relative to the validator's"
          />
          <Fact label="Miner text" value={`${formatInt(row.miner_chars)} chars`} />
          <Fact label="Validator text" value={`${formatInt(row.validator_chars)} chars`} />
          <Fact
            label="Fetched via"
            value={row.via ?? EMPTY_VALUE}
            hint="How the validator fetched the page"
          />
          <Fact label="Validator error" value={row.validator_error ?? EMPTY_VALUE} />
        </dl>
      ) : null}
      {hasWindows ? (
        <div>
          <p className="text-ink-muted mb-2 text-xs">
            The texts first differ
            {row.diff_at === null ? '' : ` at character ${formatInt(row.diff_at)}`}. The differing
            part is marked.
          </p>
          <div className="grid gap-3 md:grid-cols-2">
            <TextPane
              title="Miner, around the difference"
              text={row.miner_window}
              compareWith={row.validator_window}
            />
            <TextPane
              title="Validator, around the difference"
              text={row.validator_window}
              compareWith={row.miner_window}
            />
          </div>
        </div>
      ) : null}
      {hasSnippets ? (
        <div className="grid gap-3 md:grid-cols-2">
          <TextPane title="Miner text, start" text={row.miner_snippet} />
          <TextPane title="Validator text, start" text={row.validator_snippet} />
        </div>
      ) : null}
    </div>
  )
}
