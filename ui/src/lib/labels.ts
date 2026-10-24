import type { BudgetCause, Kind, TaskStatus, UrlOutcome, Verdict } from '@/api/types'

export const VERDICTS: readonly Verdict[] = ['pass', 'fail', 'void']

export const VERDICT_LABEL: Record<Verdict, string> = {
  pass: 'Pass',
  fail: 'Fail',
  void: 'Void',
}

export const VERDICT_MEANING: Record<Verdict, string> = {
  pass: "The upload held up when validators checked it. What it returned counts toward its miner's share.",
  fail: 'The upload did not hold up. Nothing in it counts and the task goes back to the queue.',
  void: 'No result could be reached. Nothing in it counts, the miner is not penalised and the task goes back to the queue.',
}

export const STATUS_LABEL: Record<TaskStatus, string> = {
  queued: 'Queued',
  claimed: 'Claimed',
  open: 'Uploaded',
  voting: 'Being checked',
  pass: 'Pass',
  fail: 'Fail',
  void: 'Void',
}

export const STATUS_MEANING: Record<TaskStatus, string> = {
  queued: 'Waiting in the queue for a miner to claim it.',
  claimed: 'A miner holds this task and has not uploaded its work yet.',
  open: 'The miner uploaded its work. Validators have not voted yet.',
  voting: 'Validators are checking the upload and voting.',
  pass: VERDICT_MEANING.pass,
  fail: VERDICT_MEANING.fail,
  void: VERDICT_MEANING.void,
}

const REASON_MEANING: Record<string, string> = {
  ok: 'The checked pages matched what the validator fetched.',
  coverage: 'Too many of the assigned URLs came back without a row.',
  content_mismatch: 'Too many checked pages had different text from what the validator fetched.',
  text_not_from_html: 'The uploaded text could not be reproduced from the uploaded HTML.',
  errors_not_reproducible: 'Pages the miner reported as errors loaded fine for the validator.',
  extra_rows: 'The upload contained rows for URLs that were not part of the task.',
  reported_rows: 'The upload held fewer good rows than the miner reported for it.',
  unreadable: 'The uploaded file could not be read.',
  upload_changed: 'The miner changed the uploaded file after reporting the task done.',
  unscorable: 'The validator could not finish its check. This does not count against the miner.',
  no_quorum: 'Not enough validators voted before the deadline.',
  validators_disagree: 'Validators reached different results and no majority formed.',
  upload_missing: 'The uploaded file could not be found.',
  inconclusive: 'The checked pages gave no evidence either way.',
  unverifiable: 'Too many of the checked pages could not be verified.',
  vectors_missing: 'Some texts of the task came back without a vector.',
  vectors_malformed:
    'Some vectors were malformed or repeated, or belonged to texts that were not part of the task.',
  vectors_mismatch:
    "A sampled text's vector fell below a similarity of 0.999 to the one the validator computed.",
}

const EMBED_REASON_MEANING: Record<string, string> = {
  ok: "Every sampled text's vector reached a similarity of 0.999 to the one the validator computed.",
  inconclusive: 'No sampled text could be compared.',
}

export function explainReason(reason: string, kind: Kind): string | null {
  return (
    (kind === 'embed' ? EMBED_REASON_MEANING[reason] : undefined) ?? REASON_MEANING[reason] ?? null
  )
}

export const BUDGET_CAUSE_MEANING: Record<BudgetCause, string> = {
  verified: 'An upload passed',
  claim_expired: 'A claim ran out',
  abandoned: 'A task was given back',
  verification_failed: 'An upload failed',
}

export const URL_OUTCOME_LABEL: Record<UrlOutcome, string> = {
  matched: 'Matched',
  mismatched: 'Mismatched',
  unverifiable: 'Unverifiable',
  errors_confirmed: 'Error confirmed',
  errors_unconfirmed: 'Error not confirmed',
  not_fetched: 'Not fetched',
}

export const EXCLUDED_MEANING =
  'Excluded: disagreed with too many final results, no longer receives work'

export const TAKE_BACK_MEANING: Record<string, string> = {
  check_failed:
    'A checked upload failed, so the uploads that passed unchecked since its last checked pass were taken back.',
  repeated_fails:
    'Checks failed repeatedly, so every upload that passed in the last 24 hours was taken back, checked or not.',
}
