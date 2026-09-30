import { useSearchParams } from 'react-router'
import { useVotes } from '@/api/queries/votes'
import { Card, CardHeader } from '@/components/ui/Card'
import { PagedList } from '@/components/ui/PagedList'
import { SegmentedControl } from '@/components/ui/SegmentedControl'
import { VotesTable } from './VotesTable'

const PARAM_VOTES = 'votes'
const VOTES_ALL = 'all'
const VOTES_DISAGREED = 'disagreed'
type VoteScope = typeof VOTES_ALL | typeof VOTES_DISAGREED

const SEGMENTS: readonly { value: VoteScope; label: string }[] = [
  { value: VOTES_ALL, label: 'All votes' },
  { value: VOTES_DISAGREED, label: 'Disagreements only' },
]

export function VoteList({ validator }: { validator: string }) {
  const [params, setParams] = useSearchParams()
  const scope: VoteScope = params.get(PARAM_VOTES) === VOTES_DISAGREED ? VOTES_DISAGREED : VOTES_ALL
  const query = useVotes(scope === VOTES_DISAGREED ? { validator, agreed: false } : { validator })

  const changeScope = (next: VoteScope) => {
    setParams(next === VOTES_DISAGREED ? { [PARAM_VOTES]: VOTES_DISAGREED } : {}, { replace: true })
  }

  return (
    <Card>
      <CardHeader
        title="Its votes"
        description="Every vote on a finalized upload, newest first"
        actions={
          <SegmentedControl
            label="Which votes to show"
            segments={SEGMENTS}
            value={scope}
            onChange={changeScope}
          />
        }
      />
      <PagedList
        query={query}
        itemsOf={(page) => page.votes}
        emptyTitle={
          scope === VOTES_DISAGREED
            ? 'This validator has not disagreed with a final result.'
            : 'This validator has not voted yet.'
        }
      >
        {(votes) => <VotesTable votes={votes} />}
      </PagedList>
    </Card>
  )
}
