export interface SplitText {
  same: string
  different: string
}

export function splitAtDifference(text: string, other: string): SplitText {
  const limit = Math.min(text.length, other.length)
  let index = 0
  while (index < limit && text[index] === other[index]) index += 1
  return { same: text.slice(0, index), different: text.slice(index) }
}
