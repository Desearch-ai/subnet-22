export type Verdict = 'pass' | 'fail' | 'void'
export type Kind = 'crawl' | 'embed'
export type TaskStatus = 'queued' | 'claimed' | 'open' | 'voting' | Verdict

export interface VerdictCounts {
  pass: number
  fail: number
  void: number
}

export interface Overview {
  as_of: number
  window_hours: number
  queue: Record<Kind, number>
  claimed: number
  validating: number
  oldest_validation_s: number | null
  publishing: number
  miners: number
  validators: { active: number; known: number }
  window: VerdictCounts & {
    tasks: number
    returned: number
    missing: number
    credited: number
    votes: number
    disagreements: number
  }
  total: VerdictCounts
}

export interface TaskSeriesPoint extends VerdictCounts {
  at: number
  tasks: number
  returned: number
  credited: number
}

export interface Series {
  bucket_s: number
  points: TaskSeriesPoint[]
}

export interface LiveClaim {
  task_id: string
  kind: Kind
  miner: string
  miner_uid: number | null
  urls: number
  expires_at: number
}

export interface Voter {
  hotkey: string
  uid: number | null
}

export interface LiveUpload {
  task_id: string
  kind: Kind
  miner: string
  miner_uid: number | null
  urls: number
  completed_at: number
  deadline: number
  voters: Voter[]
  electorate: number
}

export interface Live {
  claims: LiveClaim[]
  uploads: LiveUpload[]
}

export interface MinerSummary extends VerdictCounts {
  hotkey: string
  uid: number | null
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
  window: VerdictCounts & { tasks: number; returned: number; credited: number }
  transitions: BudgetTransition[]
}

export interface ValidatorSummary extends VerdictCounts {
  hotkey: string
  uid: number | null
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
  coldkey: string | null
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
  validator: string
  validator_uid: number | null
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
  validator: string
  validator_uid: number | null
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

export interface TaskDetail {
  task_id: string
  status: TaskStatus
  round_id: string | null
  miner: string | null
  miner_uid: number | null
  score: TaskScore | null
  uploads: TaskScore[]
  votes: Vote[]
  urls: TaskUrl[]
}
