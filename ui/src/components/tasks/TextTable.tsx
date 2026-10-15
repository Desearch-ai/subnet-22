import type { TaskText } from '@/api/types'
import { ScrollValue } from '@/components/ui/ScrollValue'
import { Table, Td, Th, Tr } from '@/components/ui/Table'
import { EMPTY_VALUE, formatSimilarity, formatSimilarityMargin } from '@/lib/format'
import { isBelowBar, MATCH_SIMILARITY, sortTexts, textName } from '@/lib/texts'
import { OutcomeBadge } from './OutcomeBadge'

export function TextTable({ texts }: { texts: readonly TaskText[] }) {
  return (
    <Table>
      <thead>
        <tr>
          <Th hint="The page's own text, or one passage of a long article">Text</Th>
          <Th>Page</Th>
          <Th
            align="right"
            hint="Cosine similarity of the miner's vector to the one the validator computed"
          >
            Similarity
          </Th>
          <Th
            align="right"
            hint={`Similarity minus ${MATCH_SIMILARITY}, the least each checked text needs. Below zero fails the upload`}
          >
            Vs {MATCH_SIMILARITY}
          </Th>
          <Th hint="What the validator found when it embedded the text again">Check</Th>
        </tr>
      </thead>
      <tbody>
        {sortTexts(texts).map((text) => {
          const tone = isBelowBar(text.similarity) ? 'text-fail' : undefined
          return (
            <Tr key={text.text_id}>
              <Td className="font-mono text-xs" title={text.text_id}>
                {textName(text)}
              </Td>
              <Td className="max-w-[14rem] md:max-w-xs lg:max-w-md xl:max-w-2xl">
                {text.url === null ? (
                  <span className="text-ink-faint">{EMPTY_VALUE}</span>
                ) : (
                  <ScrollValue>
                    <a
                      href={text.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="hover:text-accent hover:underline"
                    >
                      {text.url}
                    </a>
                  </ScrollValue>
                )}
              </Td>
              <Td numeric className={tone}>
                {formatSimilarity(text.similarity)}
              </Td>
              <Td numeric className={tone ?? 'text-ink-muted'}>
                {formatSimilarityMargin(text.similarity, MATCH_SIMILARITY)}
              </Td>
              <Td>
                <OutcomeBadge outcome={text.outcome} />
              </Td>
            </Tr>
          )
        })}
      </tbody>
    </Table>
  )
}
