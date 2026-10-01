import { Fragment, useState } from 'react'
import type { TaskUrl, UrlOutcome } from '@/api/types'
import { Badge, type Tone } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { ScrollValue } from '@/components/ui/ScrollValue'
import { SegmentedControl } from '@/components/ui/SegmentedControl'
import { EmptyState } from '@/components/ui/States'
import { Table, Td, Th } from '@/components/ui/Table'
import { cn } from '@/lib/cn'
import { EMPTY_VALUE, formatInt, formatScore } from '@/lib/format'
import { URL_OUTCOME_LABEL } from '@/lib/labels'
import { UrlDetail } from './UrlDetail'

type UrlFilter = 'all' | 'checked' | 'mismatched' | 'errors'

const OUTCOME_TONE: Record<UrlOutcome, Tone> = {
  matched: 'pass',
  mismatched: 'fail',
  unverifiable: 'void',
  errors_confirmed: 'neutral',
  errors_unconfirmed: 'fail',
  not_fetched: 'neutral',
}

const FILTER_TEST: Record<UrlFilter, (row: TaskUrl) => boolean> = {
  all: () => true,
  checked: (row) => row.sampled,
  mismatched: (row) => row.outcome === 'mismatched',
  errors: (row) =>
    row.error !== null ||
    row.outcome === 'errors_confirmed' ||
    row.outcome === 'errors_unconfirmed',
}

const FILTER_LABEL: Record<UrlFilter, string> = {
  all: 'All',
  checked: 'Checked',
  mismatched: 'Mismatched',
  errors: 'Errors',
}

const FILTERS: readonly UrlFilter[] = ['checked', 'mismatched', 'errors', 'all']
const COLUMN_COUNT = 8
const PAGE_ROWS = 50

interface UrlRowProps {
  row: TaskUrl
  open: boolean
  onToggle: () => void
}

function UrlRow({ row, open, onToggle }: UrlRowProps) {
  return (
    <Fragment>
      <tr className={cn('hover:bg-raised/60 transition-colors', open ? 'bg-raised/60' : null)}>
        <Td className="w-8 pr-0">
          {row.sampled ? (
            <button
              type="button"
              onClick={onToggle}
              aria-expanded={open}
              aria-label={open ? 'Hide details' : 'Show details'}
              className="text-ink-muted hover:bg-raised hover:text-ink border-line-strong size-5 rounded-sm border font-mono text-xs leading-none"
            >
              {open ? '−' : '+'}
            </button>
          ) : null}
        </Td>
        <Td className="max-w-[14rem] md:max-w-xs lg:max-w-md xl:max-w-2xl">
          <ScrollValue>
            <a
              href={row.url}
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-accent hover:underline"
            >
              {row.url}
            </a>
          </ScrollValue>
        </Td>
        <Td numeric>{row.status ?? EMPTY_VALUE}</Td>
        <Td className="font-mono text-xs">
          {row.error ?? <span className="text-ink-faint">{EMPTY_VALUE}</span>}
        </Td>
        <Td numeric>{formatInt(row.text_chars)}</Td>
        <Td>
          {row.outcome ? (
            <Badge tone={OUTCOME_TONE[row.outcome]}>{URL_OUTCOME_LABEL[row.outcome]}</Badge>
          ) : (
            <span className="text-ink-faint text-xs">
              {row.sampled ? EMPTY_VALUE : 'not checked'}
            </span>
          )}
        </Td>
        <Td numeric>{formatScore(row.similarity)}</Td>
        <Td>
          {row.rejected ? (
            <Badge tone="warn" title="This row is left out of the published data">
              Left out
            </Badge>
          ) : null}
        </Td>
      </tr>
      {open && row.sampled ? (
        <tr>
          <td colSpan={COLUMN_COUNT} className="border-line bg-raised/30 border-b">
            <UrlDetail row={row} />
          </td>
        </tr>
      ) : null}
    </Fragment>
  )
}

export function UrlTable({ urls }: { urls: readonly TaskUrl[] }) {
  const [filter, setFilter] = useState<UrlFilter>(
    urls.some(FILTER_TEST.checked) ? 'checked' : 'all',
  )
  const [openUrl, setOpenUrl] = useState<string | null>(null)
  const [shown, setShown] = useState(PAGE_ROWS)
  const matching = urls
    .filter(FILTER_TEST[filter])
    .toSorted((one, other) => Number(other.sampled) - Number(one.sampled))
  const rows = matching.slice(0, shown)
  const segments = FILTERS.map((value) => ({
    value,
    label: `${FILTER_LABEL[value]} ${urls.filter(FILTER_TEST[value]).length}`,
  }))

  return (
    <>
      <div className="border-line flex flex-wrap items-center justify-between gap-2 border-b px-4 py-2">
        <SegmentedControl
          label="Filter URLs"
          segments={segments}
          value={filter}
          onChange={(next) => {
            setFilter(next)
            setShown(PAGE_ROWS)
          }}
        />
        <span className="text-ink-faint text-xs">
          Open a checked row to see why and compare the texts
        </span>
      </div>
      {rows.length === 0 ? (
        <EmptyState title="No URL matches this filter." />
      ) : (
        <Table>
          <thead>
            <tr>
              <Th />
              <Th>URL</Th>
              <Th align="right" hint="The HTTP status written in the miner's upload for this URL">
                Status in upload
              </Th>
              <Th hint="The error written in the miner's upload for this URL, when it has no page">
                Error in upload
              </Th>
              <Th align="right" hint="Length of the text in the miner's upload">
                Text chars
              </Th>
              <Th hint="What the validator found when it fetched the page again">Check</Th>
              <Th align="right" hint="How close the two texts are, from 0 to 1">
                Similarity
              </Th>
              <Th />
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <UrlRow
                key={row.url}
                row={row}
                open={openUrl === row.url}
                onToggle={() => {
                  setOpenUrl(openUrl === row.url ? null : row.url)
                }}
              />
            ))}
          </tbody>
        </Table>
      )}
      {matching.length > rows.length ? (
        <div className="border-line flex items-center justify-between gap-2 border-t px-4 py-2">
          <span className="text-ink-faint text-xs">
            Showing {formatInt(rows.length)} of {formatInt(matching.length)}
          </span>
          <Button
            onClick={() => {
              setShown(shown + PAGE_ROWS)
            }}
          >
            Show more
          </Button>
        </div>
      ) : null}
    </>
  )
}
