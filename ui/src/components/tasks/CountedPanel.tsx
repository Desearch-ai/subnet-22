import type { TaskScore, TaskText } from '@/api/types'
import { Card, CardHeader } from '@/components/ui/Card'
import { formatInt, formatPercent, formatSimilarity, ratio } from '@/lib/format'
import { EMBED_SAMPLES, lowestSimilarity, MATCH_SIMILARITY } from '@/lib/texts'

interface StepProps {
  value: string
  label: string
  detail?: string
}

function Step({ value, label, detail }: StepProps) {
  return (
    <li className="border-line min-w-0 flex-1 basis-32 rounded-md border px-3 py-2">
      <div className="text-ink font-mono text-lg font-medium tabular-nums">{value}</div>
      <div className="text-ink-muted text-xs">{label}</div>
      {detail ? <div className="text-ink-faint text-xs">{detail}</div> : null}
    </li>
  )
}

function CrawlSteps({ score }: { score: TaskScore }) {
  const assigned = score.returned + score.missing
  const checkedErrors = score.errors_confirmed + score.errors_unconfirmed
  return (
    <>
      <ol className="flex flex-wrap gap-2">
        <Step value={formatInt(assigned)} label="URLs assigned" />
        <Step
          value={formatInt(score.returned)}
          label="rows returned"
          detail={`${formatPercent(ratio(score.returned, assigned))} of assigned`}
        />
        <Step value={formatInt(score.sampled)} label="pages checked" />
        <Step
          value={formatInt(score.matched)}
          label="checked pages matched"
          detail="sets the share of content rows counted"
        />
        <Step
          value={`${formatInt(score.errors_confirmed)} / ${formatInt(checkedErrors)}`}
          label="checked errors confirmed"
          detail="sets the share of error rows counted"
        />
        <Step
          value={formatInt(score.credited)}
          label="rows counted"
          detail={`${formatPercent(ratio(score.credited, score.returned))} of returned`}
        />
      </ol>
      <p className="text-ink-muted mt-3 text-sm">
        {score.verdict === 'pass'
          ? 'Rows with page content count at the share of checked pages that matched. Rows that reported an error count at the share of checked errors the validator confirmed. A mismatch in the sample therefore lowers the counted rows for the whole upload, not just for that page.'
          : 'No rows count for an upload that failed or ended void.'}
      </p>
    </>
  )
}

function EmbedSteps({ score, texts }: { score: TaskScore; texts: readonly TaskText[] }) {
  const assigned = score.returned + score.missing
  return (
    <>
      <ol className="flex flex-wrap gap-2">
        <Step value={formatInt(assigned)} label="texts assigned" />
        <Step
          value={formatInt(score.returned)}
          label="texts returned"
          detail={`${formatPercent(ratio(score.returned, assigned))} of assigned`}
        />
        <Step
          value={`${formatInt(score.matched)} / ${formatInt(score.sampled)}`}
          label="checked texts matched"
          detail={`each needs ${MATCH_SIMILARITY}`}
        />
        <Step
          value={formatSimilarity(lowestSimilarity(texts))}
          label="lowest similarity"
          detail={texts.length === 0 ? 'no checked texts kept' : 'of the checked texts'}
        />
        <Step value={formatInt(score.credited)} label="characters counted" />
      </ol>
      <p className="text-ink-muted mt-3 text-sm">
        {score.verdict !== 'pass'
          ? 'No characters count for an upload that failed or ended void.'
          : score.sampled === 0
            ? "Not drawn for a check, so all characters of the task count. A later failed check of this miner's embed uploads takes them back."
            : 'Every checked text reached the bar, so all characters of the task count. One checked text below it fails the whole upload.'}
      </p>
    </>
  )
}

interface CountedPanelProps {
  score: TaskScore
  texts: readonly TaskText[]
}

export function CountedPanel({ score, texts }: CountedPanelProps) {
  const embed = score.kind === 'embed'
  return (
    <Card>
      <CardHeader
        title={
          embed ? 'From checked texts to characters counted' : 'From checked pages to rows counted'
        }
        description={
          embed
            ? `Validators embed ${EMBED_SAMPLES} sampled texts again; each must reach a similarity of ${MATCH_SIMILARITY} to the miner's vector.`
            : "Validators fetch a sample of the pages again. What holds up in the sample decides how many of the returned rows count toward the miner's share."
        }
      />
      <div className="p-4">
        {embed ? <EmbedSteps score={score} texts={texts} /> : <CrawlSteps score={score} />}
      </div>
    </Card>
  )
}
