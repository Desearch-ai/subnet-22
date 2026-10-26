export type Verdict = 'pass' | 'fail' | 'void'
export type Kind = 'crawl' | 'embed'
export type TaskStatus = 'queued' | 'claimed' | 'open' | 'voting' | Verdict

export interface VerdictCounts {
  pass: number
  fail: number
  void: number
}

export interface ClaimCounts {
  issued: number
  completed: number
  lapsed: number
  requeued: number
  dropped: number
  refused: Partial<Record<string, number>>
}

export interface WaitSummary {
  tasks: number
  p50: number | null
  p90: number | null
}

export interface VerdictWaits {
  unchecked: WaitSummary | null
  checked: WaitSummary | null
}

export interface ReasonCount {
  verdict: Verdict
  reason: string
  checked: boolean
  tasks: number
}

export interface TakenBack {
  take_backs: number
  tasks: number
  miners: number
  credited: number
}

export interface Overview {
  as_of: number
  window_hours: number
  kind: Kind
  queue: Record<Kind, number>
  claimed: number
  validating: number
  oldest_validation_s: number | null
  publishing: number
  miners: number
  locked_out: number
  validators: { active: number; known: number; agreement: number | null }
  window: VerdictCounts & {
    tasks: number
    returned: number
    missing: number
    credited: number
    votes: number
    disagreements: number
  }
  claims: ClaimCounts
  checks: { checked: number; verdict_s: VerdictWaits; reasons: ReasonCount[] }
  taken_back: TakenBack
  total: VerdictCounts
}

export interface HourPoint extends VerdictCounts, ClaimCounts {
  at: number
  tasks: number
  returned: number
  credited: number
  miners?: number
  checked?: number
  verdict_s?: VerdictWaits | null
}

export interface Hours {
  kind: Kind
  hours: number
  miner: string | null
  as_of: number
  points: HourPoint[]
}

export interface MinerSummary extends VerdictCounts {
  hotkey: string
  uid: number | null
  coldkey: string | null
  budget: number
  in_flight: number
  waiting: number
  locked_until: number | null
  verified: number
  tasks: number
  returned: number
  credited: number
  coverage: number | null
  share: number
  last_scored_at: number | null
}

export interface MinersResponse {
  window_hours: number
  kind: Kind
  miners: MinerSummary[]
}

export interface MinerPool {
  budget: number
  verified: number
  in_flight: number
  waiting: number
  locked_until: number | null
}

export interface MinerCoverage {
  assigned: number
  returned: number
  coverage: number | null
}

export interface MinerWindow extends VerdictCounts {
  tasks: number
  returned: number
  missing: number
  credited: number
}

export interface MinerKind {
  window: MinerWindow
  share: number
  claims: ClaimCounts
}

export interface TakeBack {
  kind: Kind
  reason: string
  tasks: number
  credited: number
  at: number
}

export type BudgetCause = 'verified' | 'claim_expired' | 'abandoned' | 'verification_failed'

export interface BudgetTransition {
  pool: Kind
  old: number
  new: number
  cause: BudgetCause
  task_id: string | null
  at: number
}

export interface MinerDetail {
  hotkey: string
  uid: number | null
  coldkey: string | null
  known: boolean
  window_hours: number
  pools: Record<Kind, MinerPool>
  coverage: Partial<MinerCoverage>
  verdicts: Partial<VerdictCounts>
  share: number
  window: MinerWindow
  kinds: Record<Kind, MinerKind>
  transitions: BudgetTransition[]
  take_backs: TakeBack[]
}

export interface ValidatorSummary extends VerdictCounts {
  hotkey: string
  uid: number | null
  coldkey: string | null
  active: boolean
  last_seen: number | null
  votes: number
  agreed: number
  disagreed: number
  agreement: number | null
  decided: number
  audits: number
  disagreements: number
  excluded: boolean
  last_vote_at: number | null
}

export interface ValidatorDetail extends ValidatorSummary {
  known: boolean
}

export interface ValidatorsResponse {
  window_hours: number
  validators: ValidatorSummary[]
}

export interface Vote {
  task_id: string
  kind: Kind
  miner: string
  miner_uid: number | null
  miner_coldkey: string | null
  validator: string
  validator_uid: number | null
  validator_coldkey: string | null
  verdict: Verdict
  reason: string
  returned: number
  sampled: number
  matched: number
  mismatched: number
  unverifiable: number
  errors_confirmed: number
  errors_unconfirmed: number
  credited: number
  final_verdict: Verdict
  final_credited: number
  agreed: boolean | null
  decided: boolean
  voted_at: number
  finalized_at: number
}

export interface VotesResponse {
  votes: Vote[]
  next: number | null
}

export interface TaskScore {
  task_id: string
  kind: Kind
  round_id: string
  miner: string
  miner_uid: number | null
  miner_coldkey: string | null
  validator: string
  validator_uid: number | null
  validator_coldkey: string | null
  verdict: Verdict
  reason: string
  returned: number
  missing: number
  duplicates: number
  sampled: number
  matched: number
  mismatched: number
  unverifiable: number
  errors_confirmed: number
  errors_unconfirmed: number
  reextract_mismatch: number
  credited: number
  upload_key: string | null
  page_key: string | null
  report_key: string | null
  claimed_at: number | null
  completed_at: number | null
  scored_at: number
}

export interface TasksResponse {
  tasks: TaskScore[]
  next: number | null
}

export type UrlOutcome =
  | 'matched'
  | 'mismatched'
  | 'unverifiable'
  | 'errors_confirmed'
  | 'errors_unconfirmed'
  | 'not_fetched'

export interface TaskUrl {
  url: string
  status: number | null
  error: string | null
  text_chars: number | null
  sampled: boolean
  outcome: UrlOutcome | null
  why: string | null
  similarity: number | null
  precision: number | null
  recall: number | null
  growth: number | null
  validator_error: string | null
  via: string | null
  miner_chars: number | null
  validator_chars: number | null
  miner_snippet: string | null
  validator_snippet: string | null
  diff_at: number | null
  miner_window: string | null
  validator_window: string | null
  rejected: boolean
}

export type TextOutcome = 'matched' | 'mismatched' | 'malformed' | 'unverifiable'

export interface TaskText {
  text_id: string
  kind: 'page' | 'passage'
  index: number
  url: string | null
  outcome: TextOutcome
  similarity: number | null
}

export interface TaskDetail {
  task_id: string
  kind: Kind
  status: TaskStatus
  round_id: string | null
  miner: string | null
  miner_uid: number | null
  miner_coldkey: string | null
  score: TaskScore | null
  uploads: TaskScore[]
  votes: Vote[]
  urls: TaskUrl[]
  texts: TaskText[]
}
