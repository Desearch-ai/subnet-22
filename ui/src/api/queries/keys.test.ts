import { describe, expect, it } from 'vitest'
import { keepsTaskList, queryKeys } from './keys'

describe('queryKeys', () => {
  it('keeps each kind of the kind-specific views apart', () => {
    expect(queryKeys.overview('crawl')).not.toEqual(queryKeys.overview('embed'))
    expect(queryKeys.miners('crawl')).not.toEqual(queryKeys.miners('embed'))
    expect(queryKeys.series({ kind: 'crawl' })).not.toEqual(queryKeys.series({ kind: 'embed' }))
    expect(queryKeys.tasks({ kind: 'crawl' }, 50)).not.toEqual(
      queryKeys.tasks({ kind: 'embed' }, 50),
    )
  })

  it('gives the whole subnet and one miner different series', () => {
    expect(queryKeys.series({ kind: 'embed' })).toEqual(['series', 'embed', null])
    expect(queryKeys.series({ kind: 'embed', miner: '5F' })).toEqual(['series', 'embed', '5F'])
  })
})

describe('keepsTaskList', () => {
  it('keeps the list up while only the result filter changes', () => {
    expect(
      keepsTaskList(
        { miner: '5F', kind: 'embed' },
        { miner: '5F', kind: 'embed', verdict: 'fail' },
      ),
    ).toBe(true)
  })

  it('drops the list when the kind, miner or validator changes, or nothing was shown before', () => {
    expect(keepsTaskList({ kind: 'crawl' }, { kind: 'embed' })).toBe(false)
    expect(keepsTaskList({}, { kind: 'embed' })).toBe(false)
    expect(keepsTaskList({ miner: '5F' }, { miner: '5G' })).toBe(false)
    expect(keepsTaskList(undefined, {})).toBe(false)
  })
})
