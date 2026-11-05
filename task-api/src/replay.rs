//! Rebuilds the accounts from the log with the API's own rules, and names every output the API logged that its rules do not give.
//!
//! The order of entries is the API's to choose and is taken as given; so are the rule times and Redis counters an entry states. Everything
//! else is worked out again under the rules (`docs/rules.md`): a validator's vote from its signed report, a verdict from the votes and the
//! round's loadability as the log built it, whether an upload is checked from the seed block's hash, a crawl round's commitment from the
//! URLs of its signed request and its batches from that set and the seed, whether a lapse is forgiven from the logged outages, budgets,
//! strikes, lockouts, take-backs and shares from the rules, and the accounts as a whole against every snapshot's digest. With anchoring on,
//! a round must be revealed only once an anchor at or past its commitment was in a block before its seed block. A difference the API decided inside one transaction is `fraud`: the log itself
//! proves it. A check the API makes on reads that may race a concurrent write is only `suspect`.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::Path;

use anyhow::Result;
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{json, Map, Value};

use crate::budgets::{self, CRAWL, EMBED, KEEP_H, SHARE_WINDOW_H};
use crate::chain::Neuron;
use crate::journal::{self, Break};
use crate::lifecycle::{self, kind_of, text, Basis, Given, Kept};
use crate::manifest::seed_from_hash;
use crate::rounds::{self, Batch, Url};
use crate::verdicts::{build_embed_vote, build_vote, decide, Decision};
use crate::{checks, models, outages, proofs, registry, rules, snapshot, validations};

/// Wrong reports are kept a week, as the API prunes them.
const WRONG_REPORTS_KEPT_S: f64 = 7.0 * 86_400.0;

/// The log itself shows the API did not follow its rules.
pub const FRAUD: &str = "fraud";
/// Worth a look, but the API reads this outside the transaction that logs it, so a race can explain it.
pub const SUSPECT: &str = "suspect";

/// One output the rules do not give.
#[derive(Clone, Debug, PartialEq)]
pub struct Divergence {
    pub seq: i64,
    pub kind: String,
    pub severity: &'static str,
    pub what: String,
    pub api: Value,
    pub replay: Value,
}

impl Divergence {
    pub fn to_json(&self) -> Value {
        json!({"seq": self.seq, "kind": self.kind, "severity": self.severity, "what": self.what, "api": self.api, "replay": self.replay})
    }
}

/// A block hash an entry used, for checking against a node of one's own.
#[derive(Clone, Debug, PartialEq)]
pub struct BlockUsed {
    pub seq: i64,
    pub block: i64,
    pub hash: String,
}

pub struct Replay {
    pub conn: Connection,
    pub divergences: Vec<Divergence>,
    pub blocks: Vec<BlockUsed>,
}

type Blobs<'a> = &'a dyn Fn(&str) -> Option<Vec<u8>>;

/// A round not revealed yet: its seed block, the entry its commitment is in, and what it committed to: a crawl round's URL set and
/// task size, an embed round's manifest.
struct Committed {
    seed_block: i64,
    commit_seq: i64,
    set: Option<(Vec<Url>, usize)>,
    manifest: Option<Vec<Value>>,
}

fn create(conn: &Connection) -> Result<()> {
    snapshot::create(conn)?;
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS replay_meta (name TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS replay_jobs (task_id TEXT PRIMARY KEY, job TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS replay_votes (task_id TEXT NOT NULL, validator TEXT NOT NULL, vote TEXT NOT NULL, PRIMARY KEY (task_id, validator));
        CREATE TABLE IF NOT EXISTS replay_claims (task_id TEXT PRIMARY KEY, hotkey TEXT NOT NULL, kind TEXT NOT NULL, state TEXT NOT NULL, at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS replay_blocks (block INTEGER PRIMARY KEY, hash TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS replay_batches (batch_id TEXT PRIMARY KEY, round_id TEXT NOT NULL, urls_hash TEXT);
        CREATE TABLE IF NOT EXISTS replay_rounds (
            round_id TEXT PRIMARY KEY, kind TEXT NOT NULL, seed_block INTEGER NOT NULL, commit_seq INTEGER NOT NULL,
            task_urls INTEGER, urls TEXT, manifest TEXT
        );
        CREATE TABLE IF NOT EXISTS replay_anchors (seq INTEGER PRIMARY KEY, block INTEGER NOT NULL);
        ",
    )?;
    Ok(())
}

fn float(value: &Value) -> f64 {
    value.as_f64().unwrap_or(0.0)
}

fn int(value: &Value) -> i64 {
    value.as_i64().or_else(|| value.as_f64().map(|x| x as i64)).unwrap_or(0)
}

impl Replay {
    /// A replay keeping its state in `path`, so it can go on from where it stopped; in memory without one.
    pub fn open(path: Option<&Path>) -> Result<Replay> {
        let conn = match path {
            Some(path) => Connection::open(path)?,
            None => Connection::open_in_memory()?,
        };
        create(&conn)?;
        Ok(Replay { conn, divergences: Vec::new(), blocks: Vec::new() })
    }

    pub fn meta(&self, name: &str) -> Result<Option<Value>> {
        let found: Option<String> = self.conn.query_row("SELECT value FROM replay_meta WHERE name = ?", [name], |row| row.get(0)).optional()?;
        Ok(found.and_then(|raw| serde_json::from_str(&raw).ok()))
    }

    fn set_meta(&self, name: &str, value: &Value) -> Result<()> {
        self.conn.execute("INSERT OR REPLACE INTO replay_meta (name, value) VALUES (?, ?)", params![name, value.to_string()])?;
        Ok(())
    }

    /// The last entry replayed and its hash.
    pub fn position(&self) -> Result<(i64, String)> {
        let last = self.meta("last")?.and_then(|v| v.as_i64()).unwrap_or(0);
        let head = self.meta("head")?.and_then(|v| v.as_str().map(String::from)).unwrap_or_else(|| journal::GENESIS.into());
        Ok((last, head))
    }

    /// The latest start entry's input: the settings and rules the copy that logs now runs with.
    fn start(&self) -> Result<Value> {
        Ok(self.meta("starts")?.and_then(|s| s.as_array().and_then(|s| s.last().cloned())).unwrap_or(Value::Null))
    }

    fn params(&self) -> Result<rules::Params> {
        Ok(serde_json::from_value(self.start()?["settings"]["rules"].clone()).unwrap_or_default())
    }

    fn anchoring(&self) -> Result<bool> {
        Ok(self.start()?["settings"]["anchored"] == true)
    }

    /// Whether a snapshot was loaded; entries before the first are only checked for their links and signatures.
    pub fn started(&self) -> Result<bool> {
        Ok(self.meta("started")?.and_then(|v| v.as_bool()).unwrap_or(false))
    }

    fn diverge(&mut self, entry: &Value, severity: &'static str, what: impl Into<String>, api: Value, replay: Value) {
        let divergence = Divergence { seq: int(&entry["seq"]), kind: text(entry, "kind").to_string(), severity, what: what.into(), api, replay };
        self.divergences.push(divergence);
    }

    fn compare(&mut self, entry: &Value, what: &str, api: &Value, replay: Value) {
        if *api != replay {
            self.diverge(entry, FRAUD, what, api.clone(), replay);
        }
    }

    /// Replays one published file: checked first for its signer and its links from where the replay stands. A replay that has
    /// replayed nothing yet may start at any file when `join` is set, trusting its `prev` until an anchor on chain confirms it.
    pub fn apply_segment(&mut self, segment: &Value, signers: &[String], join: bool, blobs: Blobs) -> Result<Result<(), Break>> {
        let (mut last, mut head) = self.position()?;
        if last == 0 && join {
            last = int(&segment["first"]) - 1;
            head = segment["prev"].as_str().unwrap_or_default().to_string();
        }
        // Any holder of the log may have signed it; which one held it when is the custody chain's to say.
        let checked: Vec<Result<(i64, String), Break>> = signers.iter().map(|signer| journal::verify_segment(segment, signer, last, &head)).collect();
        if !checked.iter().any(Result::is_ok) {
            // A file one holder did sign but that breaks the links says more than a signature no holder made.
            let number = segment["segment"].as_i64().unwrap_or(-1);
            let broken =
                checked.into_iter().filter_map(Result::err).find(|b| !matches!(b, Break::Signature { .. })).unwrap_or(Break::Signature { segment: number });
            return Ok(Err(broken));
        }
        for entry in segment["entries"].as_array().into_iter().flatten() {
            self.apply(entry, blobs)?;
        }
        Ok(Ok(()))
    }

    /// Replays one entry whose links were checked.
    pub fn apply(&mut self, entry: &Value, blobs: Blobs) -> Result<()> {
        let kind = text(entry, "kind").to_string();
        if let Some(request) = entry["input"].get("request") {
            if journal::request_body(request, blobs).is_none() {
                self.diverge(entry, FRAUD, "the request body this entry names was not published", request.clone(), Value::Null);
            } else if !journal::request_signed(request, blobs) {
                self.diverge(entry, FRAUD, "the request is not signed by the hotkey it names", request.clone(), Value::Null);
            }
        }
        self.receipts_name_the_request(entry);
        let accepted = entry["output"].get("refused").is_none();
        if accepted && matches!(kind.as_str(), "complete" | "abandon" | "report" | "settle" | "finalize") && self.meta("paused")? == Some(true.into()) {
            self.diverge(entry, SUSPECT, "the accounts changed while the log shows a pause", entry["input"]["task_id"].clone(), Value::Null);
        }
        if kind == "snapshot" {
            self.snapshot(entry, blobs)?;
        } else if kind == "start" {
            let mut starts: Vec<Value> = self.meta("starts")?.and_then(|s| s.as_array().cloned()).unwrap_or_default();
            starts.push(entry["input"].clone());
            // A start older than any claim still open no longer matters.
            if starts.len() > 32 {
                starts.remove(0);
            }
            self.set_meta("starts", &json!(starts))?;
        } else if kind == "custody" {
            self.custody(entry)?;
        } else if kind == "started" {
            // A copy's start lowered to the rule time of its first block: the settings stay those of its start entry.
            let mut starts: Vec<Value> = self.meta("starts")?.and_then(|s| s.as_array().cloned()).unwrap_or_default();
            if let Some(mut lowered) = starts.last().cloned() {
                lowered["started_at"] = entry["input"]["started_at"].clone();
                starts.push(lowered);
                self.set_meta("starts", &json!(starts))?;
            }
        } else if kind == "clock" {
            if let Some(before) = self.meta("clock")?.filter(|before| *before != entry["input"]) {
                self.diverge(entry, FRAUD, "the clock every rule time is counted from was anchored again", entry["input"].clone(), before);
            }
            self.set_meta("clock", &entry["input"])?;
        } else if kind == "pause" {
            self.set_meta("paused", &entry["input"]["paused"])?;
            // What a CVM appointed without a handover holds as the pause, should the log stop paused.
            let pause = json!({"reason": entry["input"]["reason"], "since": entry["at"], "by": entry["input"]["request"]["hotkey"]});
            self.set_meta("pause", &pause)?;
        } else if kind == "anchor" {
            self.conn.execute(
                "INSERT OR REPLACE INTO replay_anchors (seq, block) VALUES (?, ?)",
                params![int(&entry["input"]["seq"]), int(&entry["input"]["block"])],
            )?;
        } else if kind == "enqueue" || kind == "embed_round" {
            // Rounds are followed from the first entry, so a reveal is checked against the commitment and its anchor.
            self.round(entry, blobs)?;
        } else if kind == "recommit" {
            self.recommit(entry, blobs)?;
        } else if kind == "reveal" {
            self.reveal(entry, blobs)?;
        } else if self.started()? {
            match kind.as_str() {
                "claim" => self.claim(entry)?,
                "miner" => {
                    let budget = budgets::get_or_create(&self.conn, text(&entry["input"], "hotkey"), text(&entry["input"], "pool"))?.budget;
                    self.compare(entry, "a new miner's budget", &entry["output"]["budget"], budget.into());
                }
                "complete" => self.complete(entry, blobs)?,
                "abandon" | "lapse" => self.lapse(entry)?,
                "report" => self.report(entry, blobs)?,
                "settle" => self.settle(entry)?,
                "finalize" => self.finalize(entry)?,
                "release" => self.close_upload(text(&entry["input"], "task_id"))?,
                "take_back" => self.take_back(entry)?,
                "recheck" => self.recheck(entry)?,
                "outage" => self.outage(entry)?,
                "prune" => {
                    let at = float(&entry["at"]);
                    budgets::prune(&self.conn, KEEP_H, at)?;
                    checks::prune(&self.conn, at)?;
                    outages::prune(&self.conn, at)?;
                    rules::prune_wrong_reports(&self.conn, at - WRONG_REPORTS_KEPT_S)?;
                }
                "shares" => {
                    let pools = budgets::shares(&self.conn, SHARE_WINDOW_H, float(&entry["input"]["as_of"]))?;
                    let api = entry["output"]["pools"].clone();
                    self.compare(entry, "the shares", &api, serde_json::to_value(pools)?);
                }
                "metagraph" => self.metagraph(entry, blobs)?,
                _ => {}
            }
        }
        self.set_meta("last", &entry["seq"])?;
        self.set_meta("head", &journal::hash_of(entry).into())?;
        Ok(())
    }

    fn snapshot(&mut self, entry: &Value, blobs: Blobs) -> Result<()> {
        let at = float(&entry["input"]["at"]);
        let api = entry["output"]["digest"].clone();
        if self.started()? {
            let ours = snapshot::digest(&snapshot::take(&self.conn, at)?);
            if api != ours.as_str() {
                let found = snapshot_body(entry, blobs);
                let detail = found.map(|theirs| differing_tables(&theirs, &snapshot::take(&self.conn, at).unwrap_or(Value::Null))).unwrap_or_default();
                self.diverge(entry, FRAUD, format!("the accounts differ from the snapshot in {}", detail.join(", ")), api, ours.into());
            }
            return Ok(());
        }
        let Some(found) = snapshot_body(entry, blobs) else {
            self.diverge(entry, FRAUD, "the snapshot this entry names was not published", api, Value::Null);
            return Ok(());
        };
        if api != snapshot::digest(&found).as_str() {
            self.diverge(entry, FRAUD, "the snapshot does not hash to its digest", api, snapshot::digest(&found).into());
            return Ok(());
        }
        snapshot::load(&self.conn, &found)?;
        self.set_meta("started", &true.into())
    }

    /// A new holder of the log: signed by whoever must have signed it. One appointed without a handover rebuilt the accounts by
    /// replaying the published log, so the accounts go on and the snapshot it logs next is compared, not loaded; the claims, uploads
    /// and rounds the lost holder had in flight are gone with it.
    fn custody(&mut self, entry: &Value) -> Result<()> {
        let record = &entry["input"]["record"];
        if !crate::custody::signature_holds(record) {
            self.diverge(entry, FRAUD, "a custody record not signed by who it says", record.clone(), Value::Null);
        }
        if record["kind"] == crate::custody::APPOINT {
            self.conn.execute_batch(
                "DELETE FROM replay_jobs; DELETE FROM replay_votes; DELETE FROM replay_claims; DELETE FROM replay_rounds; DELETE FROM replay_batches;",
            )?;
        }
        Ok(())
    }

    fn claim(&mut self, entry: &Value) -> Result<()> {
        // A claim states the rule time it was judged at; a refusal its wall time, and a lockout's its rule time and wait.
        let at = entry["input"]["at"].as_f64().unwrap_or_else(|| float(&entry["at"]));
        let hotkey = entry["input"]["request"]["hotkey"].as_str().unwrap_or_default().to_string();
        let kind = entry["input"].get("kind").and_then(Value::as_str).unwrap_or(CRAWL).to_string();
        for receipt in entry["output"]["receipts"].as_array().into_iter().flatten() {
            let body = &receipt["body"];
            if body["hotkey"] != hotkey.as_str() {
                self.diverge(entry, FRAUD, "a receipt names another hotkey than the request", body.clone(), hotkey.clone().into());
                continue;
            }
            match text(body, "outcome") {
                "issued" => {
                    if self.meta("paused")? == Some(true.into()) {
                        self.diverge(entry, SUSPECT, "a task was issued while the log shows a pause", body.clone(), Value::Null);
                    }
                    if let Some(until) = budgets::locked_until(&self.conn, &hotkey, &kind, at)? {
                        self.diverge(entry, SUSPECT, "a task was issued to a locked-out miner", body.clone(), until.into());
                    }
                    let task_id = text(body, "task_id");
                    self.conn.execute(
                        "INSERT OR REPLACE INTO replay_claims (task_id, hotkey, kind, state, at) VALUES (?, ?, ?, 'held', ?)",
                        params![task_id, hotkey, kind, at],
                    )?;
                }
                "refused" if body["refusal"]["code"] == "LOCKED_OUT" => {
                    let inputs = &body["refusal"]["inputs"];
                    let judged_at = float(&inputs["until"]) - float(&inputs["retry_after"]);
                    let locked = [CRAWL, EMBED].iter().map(|pool| budgets::locked_until(&self.conn, &hotkey, pool, judged_at)).collect::<Result<Vec<_>>>()?;
                    if locked.iter().all(Option::is_none) {
                        self.diverge(entry, SUSPECT, "a miner was refused as locked out while it was not", body.clone(), Value::Null);
                    }
                }
                "refused" if body["refusal"]["code"] == "PAUSED" && self.meta("paused")? != Some(true.into()) => {
                    self.diverge(entry, SUSPECT, "a claim was refused as paused while the log shows no pause", body.clone(), Value::Null);
                }
                _ => {}
            }
        }
        let held: i64 =
            self.conn
                .query_row("SELECT COUNT(*) FROM replay_claims WHERE hotkey = ? AND kind = ? AND state = 'held'", params![hotkey, kind], |row| row.get(0))?;
        let budget = budgets::get(&self.conn, &hotkey, &kind)?.budget;
        if held > budget {
            self.diverge(entry, SUSPECT, "the miner holds more tasks than its budget", held.into(), budget.into());
        }
        // The on-chain incentive caps the budget a claim uses, as the metagraph the log holds gives it; the API reads the work it
        // compares against from a cache, so a difference is only suspect.
        let allowed = budget.min(self.incentive_cap(&hotkey, at)?);
        if entry["input"]["budget"].as_i64().is_some_and(|used| used != allowed) {
            self.diverge(
                entry,
                SUSPECT,
                "the budget a claim used is not the one the checks and the incentive give",
                entry["input"]["budget"].clone(),
                allowed.into(),
            );
        }
        Ok(())
    }

    /// The budget a hotkey's incentive allows, as `State::budget_cap` works it out.
    fn incentive_cap(&self, hotkey: &str, at: f64) -> Result<i64> {
        let params = self.params()?;
        let Some(incentive) = self.meta("incentives")?.and_then(|known| known[hotkey].as_f64()).filter(|_| params.incentive_full > 0.0) else {
            return Ok(budgets::CEILING);
        };
        let work = budgets::work(&self.conn, SHARE_WINDOW_H, at)?;
        let Some((shares, first)) = work.get(hotkey) else { return Ok(budgets::CEILING) };
        let expected = rules::expected_incentive(shares, &params);
        if *first > budgets::hour_of(at) - params.incentive_grace_h || expected <= 0.0 {
            return Ok(budgets::CEILING);
        }
        Ok(rules::incentive_cap(Some(incentive / expected), &params))
    }

    fn complete(&mut self, entry: &Value, blobs: Blobs) -> Result<()> {
        if entry["output"].get("refused").is_some() {
            return Ok(());
        }
        let input = &entry["input"];
        let Some(job) = lifecycle::restored_job(&input["job"], blobs) else {
            self.diverge(entry, FRAUD, "the job's URLs this entry names were not published", input["job"].clone(), Value::Null);
            return Ok(());
        };
        let task_id = text(input, "task_id");
        let hotkey = input["request"]["hotkey"].as_str().unwrap_or_default();
        if text(&job, "miner") != hotkey || text(&job, "task_id") != task_id {
            self.diverge(entry, FRAUD, "the job names another miner or task than the request", job.clone(), json!({"miner": hotkey, "task_id": task_id}));
        }
        let committed: Option<Option<String>> =
            self.conn.query_row("SELECT urls_hash FROM replay_batches WHERE batch_id = ?", [task_id], |row| row.get(0)).optional()?;
        let urls_hash = proofs::sha256(&crate::py::canonical(&job["urls"]));
        if let Some(Some(committed)) = committed {
            if kind_of(&job) == CRAWL && committed != urls_hash {
                self.diverge(entry, FRAUD, "the task's URLs are not the ones its round committed to", urls_hash.into(), committed.into());
            }
        }
        self.conn.execute("DELETE FROM replay_votes WHERE task_id = ?", [task_id])?;
        self.conn.execute("INSERT OR REPLACE INTO replay_jobs (task_id, job) VALUES (?, ?)", params![task_id, job.to_string()])?;
        self.conn.execute("UPDATE replay_claims SET state = 'waiting' WHERE task_id = ? AND hotkey = ?", params![task_id, hotkey])?;
        Ok(())
    }

    fn lapse(&mut self, entry: &Value) -> Result<()> {
        let (input, at) = (&entry["input"], float(&entry["at"]));
        let task_id = text(input, "task_id");
        let hotkey = if text(entry, "kind") == "abandon" { input["request"]["hotkey"].as_str().unwrap_or_default() } else { text(input, "hotkey") };
        let kind = input.get("kind").and_then(Value::as_str).unwrap_or(CRAWL);
        if text(entry, "kind") == "lapse" {
            let waiting: Option<String> = self
                .conn
                .query_row("SELECT state FROM replay_claims WHERE task_id = ? AND hotkey = ?", params![task_id, hotkey], |row| row.get(0))
                .optional()?;
            if waiting.as_deref() == Some("waiting") {
                self.diverge(entry, FRAUD, "a claim was struck as expired after its upload was accepted", input.clone(), Value::Null);
            }
            if input["expiry"].as_f64().is_some_and(|expiry| expiry > at) {
                self.diverge(entry, FRAUD, "a claim was struck as expired before it expired", input["expiry"].clone(), at.into());
            }
        }
        let ours = self.forgiveness(entry, kind)?;
        let theirs = match &input["forgiven"] {
            Value::String(cause) => Some(cause.clone()),
            Value::Bool(true) => Some("restart".to_string()),
            _ => None,
        };
        self.compare(entry, "why the lapse costs the miner nothing", &json!(theirs), json!(ours));
        let forgiven = theirs.is_some();
        let budget = if forgiven {
            budgets::get_or_create(&self.conn, hotkey, kind)?.budget
        } else {
            let cause = if text(entry, "kind") == "abandon" { "abandoned" } else { "claim_expired" };
            lifecycle::lapse_counted(&self.conn, hotkey, task_id, kind, int(&input["assigned"]), cause, at)?
        };
        if !(text(entry, "kind") == "lapse" && forgiven) {
            self.compare(entry, "the budget after the lapse", &entry["output"]["budget"], budget.into());
        }
        self.conn.execute("DELETE FROM replay_claims WHERE task_id = ? AND hotkey = ?", params![task_id, hotkey])?;
        Ok(())
    }

    /// Why a lapse costs nothing by the rules: held through the restart of the copy that decided, which must be one the log saw
    /// start, or alive while an outage the log holds was open.
    fn forgiveness(&mut self, entry: &Value, kind: &str) -> Result<Option<String>> {
        let input = &entry["input"];
        let basis = &input["forgiving"];
        let starts: Vec<Value> = self.meta("starts")?.and_then(|s| s.as_array().cloned()).unwrap_or_default();
        let start = starts.last().cloned().unwrap_or(Value::Null);
        let claim_ttl = float(&basis["claim_ttl"]);
        if !start.is_null() {
            let ttl = float(&start["settings"][if kind == EMBED { "embed_claim_ttl" } else { "claim_ttl" }]) + crate::rounds::UPLOAD_GRACE_S;
            if claim_ttl != ttl {
                self.diverge(entry, FRAUD, "the claim time a lapse was judged on", basis["claim_ttl"].clone(), ttl.into());
            }
            if let Some(block_seconds) = start["settings"]["block_seconds"].as_f64().filter(|b| basis["block_seconds"].as_f64() != Some(*b)) {
                self.diverge(entry, FRAUD, "the block length a lapse was judged on", basis["block_seconds"].clone(), block_seconds.into());
            }
        }
        let Some(expiry) = input["expiry"].as_f64() else { return Ok(None) };
        if let Some(started_at) = basis["started_at"].as_f64() {
            if !starts.iter().any(|s| s["started_at"].as_f64() == Some(started_at)) {
                self.diverge(entry, FRAUD, "a lapse was decided by a copy the log never saw start", basis["started_at"].clone(), json!(starts));
            }
            if expiry <= started_at + claim_ttl {
                return Ok(Some("restart".into()));
            }
        }
        let from = expiry - claim_ttl - 2.0 * float(&basis["block_seconds"]);
        Ok(outages::during(&self.conn, from, expiry)?.map(|cause| format!("{cause}_outage")))
    }

    /// An outage opened or closed: one of a cause at a time, as the API keeps them.
    fn outage(&mut self, entry: &Value) -> Result<()> {
        let input = &entry["input"];
        let (cause, at, block) = (text(input, "cause"), float(&input["at"]), int(&input["block"]));
        let changed = match text(input, "event") {
            "begin" => outages::begin(&self.conn, cause, at, block, text(input, "detail"))?,
            _ => outages::end(&self.conn, cause, at, block)?,
        };
        if !changed {
            self.diverge(entry, FRAUD, "an outage began while one of its cause was open, or ended while none was", input.clone(), Value::Null);
        }
        Ok(())
    }

    /// Every receipt answering a signed request names that request, so a relaying validator's witness can be matched to it.
    fn receipts_name_the_request(&mut self, entry: &Value) {
        let Some(request) = entry["input"]["request"]["request"].as_str().map(String::from) else { return };
        for receipt in entry["output"]["receipts"].as_array().cloned().unwrap_or_default() {
            if receipt["body"]["request"] != request.as_str() {
                self.diverge(entry, FRAUD, "a receipt does not name the request it answers", receipt["body"].clone(), request.clone().into());
            }
        }
    }

    fn job(&self, task_id: &str) -> Result<Option<Value>> {
        let found: Option<String> = self.conn.query_row("SELECT job FROM replay_jobs WHERE task_id = ?", [task_id], |row| row.get(0)).optional()?;
        Ok(found.and_then(|raw| serde_json::from_str(&raw).ok()))
    }

    fn report(&mut self, entry: &Value, blobs: Blobs) -> Result<()> {
        let input = &entry["input"];
        let task_id = text(input, "task_id");
        let validator = input["request"]["hotkey"].as_str().unwrap_or_default().to_string();
        let validators = self.meta("validators")?;
        if validators.as_ref().and_then(Value::as_array).is_some_and(|known| !known.iter().any(|v| v == validator.as_str())) {
            self.diverge(entry, FRAUD, "a report was taken from a hotkey the metagraph does not let vote", validator.clone().into(), Value::Null);
        }
        let Some(job) = self.job(task_id)? else { return Ok(()) };
        let body = journal::request_body(&input["request"], blobs).and_then(|body| serde_json::from_slice::<Value>(&body).ok());
        let Some(Value::Object(body)) = body else { return Ok(()) };
        let built = if kind_of(&job) == EMBED {
            models::embed_score(&body).ok().map(|result| build_embed_vote(&job, &validator, result))
        } else {
            models::score(&body).ok().map(|result| build_vote(&job, &validator, result))
        };
        if let Some(refused) = entry["output"].get("refused") {
            let at = float(&entry["at"]);
            if refused["status"] == 422 && matches!(built, Some(Ok(_))) {
                self.diverge(entry, FRAUD, "a feasible report was refused", refused.clone(), Value::Null);
            }
            // A report its task could not have produced counts against its validator, and only such a report does.
            match (refused["status"] == 422 && matches!(built, Some(Err(_))), input["wrong_report"].as_str()) {
                (true, Some(why)) => rules::note_wrong_report(&self.conn, &validator, task_id, why, at)?,
                (true, None) => self.diverge(entry, FRAUD, "a report its task could not have produced was not counted", refused.clone(), Value::Null),
                (false, Some(_)) => self.diverge(entry, FRAUD, "a report was counted wrong that its task could have produced", refused.clone(), Value::Null),
                (false, None) => {}
            }
            let settings = &self.start()?["settings"];
            let (since, limit) = (at - float(&settings["wrong_reports_window_s"]), settings["wrong_reports"].as_i64().unwrap_or(0));
            if refused["status"] == 403 && !validations::is_excluded(&self.conn, &validator, since, limit)? {
                self.diverge(entry, FRAUD, "a report was refused from a validator that is not excluded", refused.clone(), Value::Null);
            }
            return Ok(());
        }
        let mut logged = entry["output"]["vote"].clone();
        let mut bare = logged.clone();
        if let Some(fields) = bare.as_object_mut() {
            fields.remove("at");
        }
        match built {
            Some(Ok(vote)) => self.compare(entry, "the vote built from the report", &bare, vote),
            _ => self.diverge(entry, FRAUD, "an infeasible report was taken as a vote", bare.clone(), Value::Null),
        }
        logged["validator"] = validator.clone().into();
        self.conn
            .execute("INSERT OR REPLACE INTO replay_votes (task_id, validator, vote) VALUES (?, ?, ?)", params![task_id, validator, logged.to_string()])?;
        Ok(())
    }

    fn note_block(&mut self, entry: &Value, block: i64, hash: &str) -> Result<()> {
        let known: Option<String> = self.conn.query_row("SELECT hash FROM replay_blocks WHERE block = ?", [block], |row| row.get(0)).optional()?;
        match known {
            Some(known) if known != hash => self.diverge(entry, FRAUD, format!("block {block} has two hashes in the log"), hash.into(), known.into()),
            Some(_) => {}
            None => {
                self.conn.execute("INSERT INTO replay_blocks (block, hash) VALUES (?, ?)", params![block, hash])?;
            }
        }
        self.blocks.push(BlockUsed { seq: int(&entry["seq"]), block, hash: hash.into() });
        Ok(())
    }

    fn settle(&mut self, entry: &Value) -> Result<()> {
        let (input, at) = (&entry["input"], float(&entry["at"]));
        let task_id = text(input, "task_id");
        self.note_block(entry, int(&input["seed_block"]), text(input, "block_hash"))?;
        if let Some(job) = self.job(task_id)? {
            let named = json!({"miner": input["miner"], "etag": input["etag"], "seed_block": input["seed_block"]});
            let stored = json!({"miner": job["miner"], "etag": job["etag"], "seed_block": job["seed_block"]});
            self.compare(entry, "the upload the draw names", &named, stored);
        }
        let reason = lifecycle::settle_reason(&self.conn, input, at)?;
        self.compare(entry, "whether the upload is checked", &entry["output"]["picked"], reason.into());
        if reason == Some(crate::sampling::RECHECK) {
            checks::took_recheck(&self.conn, text(input, "miner"), text(input, "kind"))?;
        }
        if let Some(mut job) = self.job(task_id)? {
            job["picked"] = reason.into();
            self.conn.execute("UPDATE replay_jobs SET job = ? WHERE task_id = ?", params![job.to_string(), task_id])?;
        }
        Ok(())
    }

    fn finalize(&mut self, entry: &Value) -> Result<()> {
        let (input, output, at) = (&entry["input"], &entry["output"], float(&entry["at"]));
        let task_id = text(input, "task_id");
        let Some(mut job) = self.job(task_id)? else {
            self.diverge(entry, FRAUD, "a verdict on an upload the log never accepted", input.clone(), Value::Null);
            return Ok(());
        };
        if text(&job, "key") != text(input, "key") {
            self.diverge(entry, FRAUD, "a verdict on another upload than the one accepted", input["key"].clone(), job["key"].clone());
        }
        job["picked"] = input["picked"].clone();
        let params = self.params()?;
        let mode = text(input, "mode").to_string();
        let (basis, decision) = match mode.as_str() {
            "checked" => {
                let voters: Vec<String> = input["voters"].as_array().into_iter().flatten().filter_map(Value::as_str).map(String::from).collect();
                let mut votes = Vec::new();
                for voter in &voters {
                    let found: Option<String> = self
                        .conn
                        .query_row("SELECT vote FROM replay_votes WHERE task_id = ? AND validator = ?", params![task_id, voter], |row| row.get(0))
                        .optional()?;
                    match found.and_then(|raw| serde_json::from_str::<Value>(&raw).ok()) {
                        Some(vote) => votes.push(vote),
                        None => self.diverge(entry, FRAUD, "a vote no validator reported", voter.clone().into(), Value::Null),
                    }
                }
                let reported: BTreeSet<String> = {
                    let mut statement = self.conn.prepare("SELECT validator FROM replay_votes WHERE task_id = ?")?;
                    let found = statement.query_map([task_id], |row| row.get::<_, String>(0))?;
                    found.collect::<rusqlite::Result<_>>()?
                };
                let left_out: Vec<&String> = reported.iter().filter(|v| !voters.contains(v)).collect();
                if !left_out.is_empty() {
                    self.diverge(entry, FRAUD, "reported votes were left out of the verdict", json!(voters), json!(left_out));
                }
                if votes.is_empty() {
                    return Ok(());
                }
                let decision = lifecycle::held_to_report(&job, decide(&votes, false, true));
                (Basis::Checked(votes), decision)
            }
            "unchecked" => {
                let vote = if kind_of(&job) == EMBED {
                    lifecycle::unchecked_embed_vote(&job)
                } else {
                    let (judged, _) = lifecycle::judge(&self.conn, &job, &Basis::Unchecked, &HashMap::new(), &params, at)?;
                    lifecycle::unchecked_vote(&job, &judged)
                };
                (Basis::Unchecked, Decision::last(vote, Vec::new()))
            }
            _ => (Basis::Unscored, Decision::last(lifecycle::unscored_vote(), Vec::new())),
        };
        // Each voter counts under its coldkey, as the metagraph the log holds names it.
        let coldkeys: HashMap<String, String> = input["coldkeys"]
            .as_object()
            .into_iter()
            .flatten()
            .map(|(hotkey, coldkey)| (hotkey.clone(), coldkey.as_str().unwrap_or_default().to_string()))
            .collect();
        let known = self.meta("coldkeys")?.unwrap_or(Value::Null);
        for (hotkey, coldkey) in &coldkeys {
            let expected = known[hotkey.as_str()].as_str().unwrap_or(hotkey.as_str());
            if !known.is_null() && coldkey != expected {
                self.diverge(entry, SUSPECT, "a voter's coldkey is not the one the metagraph names", coldkey.clone().into(), expected.into());
            }
        }
        let vote = decision.vote().clone();
        let result = vote["result"].as_object().cloned().unwrap_or_default();
        let published = text(&vote, "verdict").to_string();
        let published_reason = result.get("reason").cloned().unwrap_or_else(|| "".into());
        if input["published"] != published.as_str() || input["published_reason"] != published_reason {
            // The votes decide what is published; an unchecked upload's own counts are judged once before the transaction.
            let severity = if mode == "checked" { FRAUD } else { SUSPECT };
            let api = json!([input["published"], input["published_reason"]]);
            self.diverge(entry, severity, "what the validators' majority found", api, json!([published, published_reason]));
        }
        let report = validations::build_report(task_id, &job, text(&vote, "validator"), &result, &decision.votes, float(&output["scored_at"]));
        let candidate = lifecycle::publish_candidate(&result).then(|| json!({}));
        let given = Given { basis: &basis, coldkeys: &coldkeys, params: &params, published: &published, published_reason: &published_reason };
        let kept = Kept { report: &report, publish: candidate.as_ref(), details: None };
        let Some(settled) = lifecycle::finalize_accounts(&self.conn, task_id, &job, &decision, given, kept, at)? else {
            self.diverge(entry, FRAUD, "a second verdict on one upload", input.clone(), Value::Null);
            return Ok(());
        };
        let checked = if lifecycle_picked(&job) { lifecycle::check_accounts(&self.conn, &job, &basis, &settled.judged, &published, at)? } else { None };
        let ours = lifecycle::finalized_output(&settled, &decision, checked.as_ref());
        let names = ["verdict", "reason", "credited", "budget", "publish", "agreed", "disagreed", "taken", "penalty"];
        let pick = |from: &Value| Value::Object(names.iter().map(|name| (name.to_string(), from.get(*name).cloned().unwrap_or(Value::Null))).collect());
        self.compare(entry, "the verdict and its accounts", &pick(output), pick(&ours));
        self.close_upload(task_id)
    }

    fn close_upload(&mut self, task_id: &str) -> Result<()> {
        self.conn.execute("DELETE FROM replay_claims WHERE task_id = ?", [task_id])?;
        self.conn.execute("DELETE FROM replay_votes WHERE task_id = ?", [task_id])?;
        Ok(())
    }

    fn take_back(&mut self, entry: &Value) -> Result<()> {
        let (input, at) = (&entry["input"], float(&entry["at"]));
        let taken = lifecycle::take_back_accounts(
            &self.conn,
            text(input, "miner"),
            text(input, "kind"),
            float(&input["since"]),
            text(input, "reason"),
            input["checked_too"].as_bool().unwrap_or(false),
            at,
        )?;
        let ours: Vec<String> = taken.into_iter().map(|t| t.task_id).collect();
        self.compare(entry, "the uploads taken back", &entry["output"]["taken"], json!(ours));
        Ok(())
    }

    fn recheck(&mut self, entry: &Value) -> Result<()> {
        let miners: BTreeSet<String> =
            entry["input"]["reports"].as_array().into_iter().flatten().map(|r| text(r, "miner").to_string()).filter(|m| !m.is_empty()).collect();
        for miner in &miners {
            checks::start_recheck(&self.conn, miner, EMBED)?;
        }
        self.compare(entry, "the miners checked again", &entry["output"]["miners"], json!(miners));
        Ok(())
    }

    fn metagraph(&mut self, entry: &Value, blobs: Blobs) -> Result<()> {
        let sha = entry["input"]["neurons"]["blob"].as_str().unwrap_or_default();
        let listed: Option<Vec<Value>> = blobs(sha).filter(|body| proofs::sha256(body) == sha).and_then(|body| serde_json::from_slice(&body).ok());
        let Some(listed) = listed else {
            self.diverge(entry, FRAUD, "the metagraph this entry names was not published", entry["input"].clone(), Value::Null);
            return Ok(());
        };
        let validators: Vec<String> = listed
            .iter()
            .filter_map(|n| {
                let neuron = Neuron {
                    uid: int(&n[0]),
                    hotkey: n[1].as_str()?.into(),
                    coldkey: n[2].as_str().unwrap_or_default().into(),
                    validator_permit: n[3].as_bool().unwrap_or(false),
                    total_stake_rao: n[4].as_str().and_then(|s| s.parse().ok()).unwrap_or(0),
                    alpha_stake_rao: n[5].as_str().and_then(|s| s.parse().ok()).unwrap_or(0),
                    incentive: n[6].as_f64(),
                };
                registry::is_validator(&neuron).then_some(neuron.hotkey)
            })
            .collect();
        self.compare(entry, "who may vote", &entry["output"]["validators"], json!(validators));
        let coldkeys: Map<String, Value> = listed
            .iter()
            .filter_map(|n| Some((n[1].as_str()?.to_string(), Value::from(n[2].as_str().filter(|c| !c.is_empty()).unwrap_or(n[1].as_str()?)))))
            .collect();
        self.set_meta("coldkeys", &Value::Object(coldkeys))?;
        let incentives: Map<String, Value> = listed.iter().filter_map(|n| Some((n[1].as_str()?.to_string(), Value::from(n[6].as_f64()?)))).collect();
        self.set_meta("incentives", &Value::Object(incentives))?;
        self.set_meta("validators", &json!(validators))
    }

    /// A round's commitment: a crawl round's is worked out again from the URLs of its signed request, an embed round's from its
    /// manifest. The round is kept until its reveal, which is checked against it.
    fn round(&mut self, entry: &Value, blobs: Blobs) -> Result<()> {
        let output = &entry["output"];
        let (round_id, seed_block) = (text(output, "round_id"), int(&output["seed_block"]));
        let (task_urls, urls, manifest) = match output.get("commitment").filter(|set| !set.is_null()) {
            Some(set) => {
                let requested = journal::request_body(&entry["input"]["request"], blobs)
                    .and_then(|body| serde_json::from_slice::<Value>(&body).ok())
                    .and_then(|body| models::Enqueue::parse(&body).ok());
                let Some(requested) = requested else {
                    self.diverge(entry, FRAUD, "the round's URLs are not those of a signed request", entry["input"].clone(), Value::Null);
                    return Ok(());
                };
                let task_urls = int(&set["task_urls"]) as usize;
                if let Some(setting) = self.start()?["settings"]["task_urls"].as_i64().filter(|t| *t as usize != task_urls) {
                    self.diverge(entry, FRAUD, "the URLs a task gets", set["task_urls"].clone(), setting.into());
                }
                let Ok((_, unique)) = lifecycle::pack_round(requested.urls, seed_block, task_urls, 0.0) else {
                    self.diverge(entry, FRAUD, "a round of URLs that do not parse", entry["input"].clone(), Value::Null);
                    return Ok(());
                };
                let urls_hash = rounds::set_hash(&unique);
                let ours =
                    json!({"url_count": unique.len(), "urls_hash": urls_hash, "task_urls": task_urls, "tasks": rounds::tasks_for(unique.len(), task_urls)});
                self.compare(entry, "the URL set the round commits to", set, ours);
                let hash = proofs::set_manifest_hash(round_id, seed_block, task_urls, unique.len(), &urls_hash);
                self.compare(entry, "the round's commitment", &output["manifest_hash"], hash.into());
                (Some(task_urls as i64), Some(lifecycle::staged_set(&unique).to_string()), None)
            }
            None => {
                let sha = output["manifest"]["blob"].as_str().unwrap_or_default();
                let manifest: Option<Vec<Value>> = blobs(sha).filter(|body| proofs::sha256(body) == sha).and_then(|body| serde_json::from_slice(&body).ok());
                let Some(manifest) = manifest else {
                    self.diverge(entry, FRAUD, "the round's manifest was not published", output.clone(), Value::Null);
                    return Ok(());
                };
                let hash = proofs::manifest_hash(&manifest, seed_block);
                self.compare(entry, "the round's commitment", &output["manifest_hash"], hash.into());
                for batch in &manifest {
                    self.conn.execute(
                        "INSERT OR REPLACE INTO replay_batches (batch_id, round_id, urls_hash) VALUES (?, ?, ?)",
                        params![text(batch, "batch_id"), round_id, batch["urls_hash"].as_str()],
                    )?;
                }
                (None, None, Some(Value::Array(manifest).to_string()))
            }
        };
        self.conn.execute(
            "INSERT OR REPLACE INTO replay_rounds (round_id, kind, seed_block, commit_seq, task_urls, urls, manifest) VALUES (?, ?, ?, ?, ?, ?, ?)",
            params![round_id, text(output, "kind"), seed_block, int(&entry["seq"]), task_urls, urls, manifest],
        )?;
        Ok(())
    }

    fn kept_round(&self, round_id: &str) -> Result<Option<Committed>> {
        let committed = |row: &rusqlite::Row| -> rusqlite::Result<Committed> {
            let (task_urls, urls, manifest): (Option<i64>, Option<String>, Option<String>) = (row.get(2)?, row.get(3)?, row.get(4)?);
            let set = urls
                .and_then(|raw| serde_json::from_str::<Vec<(String, String)>>(&raw).ok())
                .map(|pairs| (pairs.into_iter().map(|(host, url)| Url { host, url }).collect(), task_urls.unwrap_or(1) as usize));
            Ok(Committed { seed_block: row.get(0)?, commit_seq: row.get(1)?, set, manifest: manifest.and_then(|raw| serde_json::from_str(&raw).ok()) })
        };
        Ok(self
            .conn
            .query_row("SELECT seed_block, commit_seq, task_urls, urls, manifest FROM replay_rounds WHERE round_id = ?", [round_id], committed)
            .optional()?)
    }

    /// Whether an anchor the log confirmed covers entry `seq` and was in a block before `seed_block`.
    fn anchored_before(&self, seq: i64, seed_block: i64) -> Result<bool> {
        Ok(self.conn.query_row("SELECT 1 FROM replay_anchors WHERE seq >= ? AND block < ?", params![seq, seed_block], |_| Ok(())).optional()?.is_some())
    }

    /// A commitment not anchored before its seed block, made again to a later one.
    fn recommit(&mut self, entry: &Value, _blobs: Blobs) -> Result<()> {
        let (input, output) = (&entry["input"], &entry["output"]);
        let round_id = text(input, "round_id");
        let Some(Committed { seed_block, commit_seq, set, manifest }) = self.kept_round(round_id)? else { return Ok(()) };
        let moved = int(&input["seed_block"]);
        if int(&input["seed_block_was"]) != seed_block || moved <= seed_block {
            self.diverge(entry, FRAUD, "a round committed again to a seed block not after its own", input.clone(), seed_block.into());
        }
        // A commitment anchored in time is revealed under its seed; committing it again would choose among seeds.
        if self.anchored_before(commit_seq, seed_block)? {
            self.diverge(entry, FRAUD, "a round anchored before its seed block was committed again", input.clone(), Value::Null);
        }
        let hash = match (&set, &manifest) {
            (Some((urls, task_urls)), _) => proofs::set_manifest_hash(round_id, moved, *task_urls, urls.len(), &rounds::set_hash(urls)),
            (None, Some(manifest)) => proofs::manifest_hash(manifest, moved),
            (None, None) => return Ok(()),
        };
        self.compare(entry, "the round's new commitment", &output["manifest_hash"], hash.into());
        self.conn.execute("UPDATE replay_rounds SET seed_block = ?, commit_seq = ? WHERE round_id = ?", params![moved, int(&entry["seq"]), round_id])?;
        Ok(())
    }

    fn reveal(&mut self, entry: &Value, blobs: Blobs) -> Result<()> {
        let (input, output) = (&entry["input"], &entry["output"]);
        let round_id = text(input, "round_id");
        self.note_block(entry, int(&input["seed_block"]), text(input, "block_hash"))?;
        let seed = seed_from_hash(text(input, "block_hash"));
        self.compare(entry, "the round's seed", &output["seed"], seed.clone().into());
        if let Some(Committed { seed_block, commit_seq, set, .. }) = self.kept_round(round_id)? {
            if int(&input["seed_block"]) != seed_block {
                self.diverge(entry, FRAUD, "a round revealed under another seed block than it committed to", input["seed_block"].clone(), seed_block.into());
            }
            if self.anchoring()? && !self.anchored_before(commit_seq, seed_block)? {
                self.diverge(entry, FRAUD, "a round was revealed before its commitment was anchored on chain", json!(commit_seq), seed_block.into());
            }
            if let Some((urls, task_urls)) = set {
                let ours: Vec<Value> = rounds::cut(&urls, &seed, round_id, task_urls).iter().map(Batch::manifest_entry).collect();
                let sha = output["batches"]["blob"].as_str().unwrap_or_default();
                let theirs: Option<Vec<Value>> = blobs(sha).filter(|body| proofs::sha256(body) == sha).and_then(|body| serde_json::from_slice(&body).ok());
                match theirs {
                    // The API lost the set it staged and the round closes empty: its fault, not a cut of its choosing.
                    Some(theirs) if theirs.is_empty() && !ours.is_empty() => {
                        self.diverge(entry, SUSPECT, "the round lost its URL set before its seed cut it", Value::Null, json!(ours.len()))
                    }
                    Some(theirs) => self.compare(entry, "the batches the seed cuts from the committed set", &json!(theirs), json!(ours.clone())),
                    None => self.diverge(entry, FRAUD, "the batches the seed cut were not published", output["batches"].clone(), Value::Null),
                }
                for batch in &ours {
                    self.conn.execute(
                        "INSERT OR REPLACE INTO replay_batches (batch_id, round_id, urls_hash) VALUES (?, ?, ?)",
                        params![text(batch, "batch_id"), round_id, batch["urls_hash"].as_str()],
                    )?;
                }
            }
            self.conn.execute("DELETE FROM replay_rounds WHERE round_id = ?", [round_id])?;
        }
        let mut statement = self.conn.prepare("SELECT batch_id FROM replay_batches WHERE round_id = ?")?;
        let ids: Vec<String> = statement.query_map([round_id], |row| row.get(0))?.collect::<rusqlite::Result<_>>()?;
        drop(statement);
        if !ids.is_empty() {
            let order = proofs::serve_order(&seed, &ids);
            self.compare(entry, "the round's serve order", &output["serve_order"], json!(order));
        }
        Ok(())
    }

    pub fn fraud(&self) -> impl Iterator<Item = &Divergence> {
        self.divergences.iter().filter(|d| d.severity == FRAUD)
    }
}

fn lifecycle_picked(job: &Value) -> bool {
    !matches!(job.get("picked"), None | Some(Value::Null) | Some(Value::Bool(false))) && job["picked"] != ""
}

fn snapshot_body(entry: &Value, blobs: Blobs) -> Option<Value> {
    let sha = entry["output"]["state"]["blob"].as_str()?;
    blobs(sha).filter(|body| proofs::sha256(body) == sha).and_then(|body| serde_json::from_slice(&body).ok())
}

fn differing_tables(theirs: &Value, ours: &Value) -> Vec<String> {
    let names: HashSet<&String> = theirs["tables"].as_object().into_iter().flat_map(|t| t.keys()).collect();
    let mut differing: Vec<String> = names.into_iter().filter(|name| theirs["tables"][name.as_str()] != ours["tables"][name.as_str()]).cloned().collect();
    differing.sort();
    differing
}

/// Every file of a log in a folder as the uploads bucket lays it out (`seq/<n>.json`, `blobs/<sha256>`), replayed in order.
pub fn replay_dir(replay: &mut Replay, dir: &Path, signers: &[String], join: bool) -> Result<Result<usize, Break>> {
    let blobs = |sha: &str| std::fs::read(dir.join("blobs").join(sha)).ok();
    let mut files: Vec<_> =
        std::fs::read_dir(dir.join("seq"))?.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "json")).collect();
    files.sort();
    let (last, _) = replay.position()?;
    let mut replayed = 0;
    for file in files {
        let segment: Value = serde_json::from_slice(&std::fs::read(&file)?)?;
        if int(&segment["last"]) <= last {
            continue;
        }
        if let Err(broken) = replay.apply_segment(&segment, signers, join, &blobs)? {
            return Ok(Err(broken));
        }
        replayed += 1;
    }
    Ok(Ok(replayed))
}

/// What an anchor on chain says: the log's hash at an entry.
pub fn check_anchor(entries: &[Value], seq: i64, anchored: &str) -> Result<(), Break> {
    match entries.iter().find(|e| int(&e["seq"]) == seq) {
        Some(entry) if journal::hash_of(entry) == anchored => Ok(()),
        Some(entry) => Err(Break::Anchor { seq, anchored: anchored.into(), found: journal::hash_of(entry) }),
        None => Err(Break::Malformed(format!("entry {seq} is not in the log read"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::Record;

    #[test]
    fn a_replay_starts_at_a_snapshot_and_checks_every_one_after() {
        let api = Connection::open_in_memory().unwrap();
        snapshot::create(&api).unwrap();
        journal::create(&api).unwrap();
        budgets::reward(&api, "m", "t0", 50, true, CRAWL, 1_800_000_000.0).unwrap();
        snapshot::record(&api, 1_800_000_000.0).unwrap();
        let at = 1_800_000_100.0;
        let lapsed = lifecycle::lapse_counted(&api, "m", "t1", CRAWL, 3, "claim_expired", at).unwrap();
        let input = json!({"task_id": "t1", "hotkey": "m", "kind": CRAWL, "assigned": 3, "expiry": at - 1.0, "forgiven": false});
        journal::append(&api, Record::new("lapse", at, input, json!({"receipts": [], "budget": lapsed}))).unwrap();
        snapshot::record(&api, at + 1.0).unwrap();
        let entries: Vec<Value> = journal::entries_after(&api, 0, 100).unwrap().into_iter().map(|(e, _)| e).collect();
        let blobs = |sha: &str| journal::blob(&api, sha).unwrap();

        let mut replay = Replay::open(None).unwrap();
        for entry in &entries {
            replay.apply(entry, &blobs).unwrap();
        }
        assert_eq!(replay.divergences, []);
        assert_eq!(budgets::get(&replay.conn, "m", CRAWL).unwrap().budget, lapsed);

        let mut lying = entries.clone();
        lying[1]["output"]["budget"] = (lapsed + 1).into();
        let mut replay = Replay::open(None).unwrap();
        for entry in &lying {
            replay.apply(entry, &blobs).unwrap();
        }
        assert_eq!(replay.divergences.len(), 1, "{:?}", replay.divergences);
        assert_eq!((replay.divergences[0].seq, replay.divergences[0].severity), (2, FRAUD));

        let mut skipped = entries.clone();
        skipped.remove(1);
        let mut replay = Replay::open(None).unwrap();
        for entry in &skipped {
            replay.apply(entry, &blobs).unwrap();
        }
        assert_eq!(replay.divergences.len(), 1, "dropping an entry shows in the next snapshot: {:?}", replay.divergences);
        assert!(replay.divergences[0].what.contains("credits"), "{}", replay.divergences[0].what);
    }
}
