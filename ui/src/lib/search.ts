import type { MinerSummary } from '@/api/types'

const HOTKEY_CHARS = 48
const DIGITS = /^\d+$/

export function looksLikeHotkey(text: string): boolean {
  return text.length === HOTKEY_CHARS && text.startsWith('5')
}

export function findMiners(
  miners: readonly MinerSummary[],
  typed: string,
  limit: number,
): MinerSummary[] {
  const text = typed.trim()
  if (!text) return []
  const byUid = DIGITS.test(text)
  const rank = (miner: MinerSummary) => (byUid && String(miner.uid) === text ? 0 : 1)
  return miners
    .filter((miner) =>
      byUid
        ? miner.uid !== null && String(miner.uid).startsWith(text)
        : miner.hotkey.includes(text),
    )
    .sort((a, b) => rank(a) - rank(b) || b.share - a.share)
    .slice(0, limit)
}
