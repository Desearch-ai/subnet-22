import { describe, expect, it } from 'vitest'
import { splitAtDifference } from './diff'

describe('splitAtDifference', () => {
  it('splits a text where it stops matching the other one', () => {
    expect(splitAtDifference('the quick fox', 'the quiet fox')).toEqual({
      same: 'the qui',
      different: 'ck fox',
    })
  })

  it('handles identical and fully different texts', () => {
    expect(splitAtDifference('same', 'same')).toEqual({ same: 'same', different: '' })
    expect(splitAtDifference('abc', 'xyz')).toEqual({ same: '', different: 'abc' })
  })
})
