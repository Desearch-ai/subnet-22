import { describe, expect, it } from 'vitest'
import { explainReason } from './labels'

describe('explainReason', () => {
  it('explains the same reason in the terms of each kind', () => {
    expect(explainReason('ok', 'crawl')).toContain('checked pages')
    expect(explainReason('ok', 'embed')).toContain('0.999')
    expect(explainReason('inconclusive', 'embed')).toBe('No sampled text could be compared.')
  })

  it('shares reasons that apply to both kinds and knows the embed ones', () => {
    expect(explainReason('no_quorum', 'embed')).toBe(explainReason('no_quorum', 'crawl'))
    expect(explainReason('vectors_mismatch', 'embed')).toContain('0.999')
    expect(explainReason('vectors_missing', 'embed')).not.toBeNull()
  })

  it('returns null for an unknown reason', () => {
    expect(explainReason('something_new', 'crawl')).toBeNull()
  })
})
