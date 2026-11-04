//! Rebuilds the accounts from the log with the API's own rules, and names every output the API logged that its rules do not give.
//!
//! The order of entries is the API's to choose and is taken as given; so are the times and Redis counters an entry states (block A moves
//! those onto block heights). Everything else is worked out again: a validator's vote from its signed report, a verdict from the votes,
//! whether an upload is checked from the seed block's hash, budgets, strikes, lockouts, take-backs and shares from the rules, and the
//! accounts as a whole against every snapshot's digest. A difference the API decided inside one transaction is `fraud`: the log itself
//! proves it. A check the API makes on reads that may race a concurrent write is only `suspect`.

use std::collections::{BTreeSet, HashSet};
use std::path::Path;

use anyhow::Result;
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{json, Map, Value};

use crate::budgets::{self, CRAWL, EMBED, KEEP_H, SHARE_WINDOW_H};
use crate::chain::Neuron;
use crate::journal::{self, Break};
use crate::lifecycle::{self, kind_of, text, Kept};
use crate::manifest::seed_from_hash;
use crate::verdicts::{build_embed_vote, build_vote, decide, Decision};
use crate::{checks, models, proofs, registry, snapshot, validations};

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

    fn meta(&self, name: &str) -> Result<Option<Value>> {
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
        } else if kind == "pause" {
            self.set_meta("paused", &entry["input"]["paused"])?;
        } else if kind == "probe" {
            crate::canary::record(&self.conn, &crate::canary::from_logged(&entry["input"]))?;
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
                "prune" => {
                    let at = float(&entry["at"]);
                    budgets::prune(&self.conn, KEEP_H, at)?;
                    checks::prune(&self.conn, at)?;
                    crate::canary::prune(&self.conn, at)?;
                }
                "shares" => {
                    let pools = budgets::shares(&self.conn, SHARE_WINDOW_H, float(&entry["input"]["as_of"]))?;
                    let api = entry["output"]["pools"].clone();
                    self.compare(entry, "the shares", &api, serde_json::to_value(pools)?);
                }
                "metagraph" => self.metagraph(entry, blobs)?,
                "enqueue" | "embed_round" => self.round(entry, blobs)?,
                "reveal" => self.reveal(entry)?,
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

    /// A new holder of the log: signed by whoever must have signed it; one appointed without a handover starts the accounts afresh,
    /// so the replay waits for the snapshot that follows.
    fn custody(&mut self, entry: &Value) -> Result<()> {
        let record = &entry["input"]["record"];
        if !crate::custody::signature_holds(record) {
            self.diverge(entry, FRAUD, "a custody record not signed by who it says", record.clone(), Value::Null);
        }
        if record["kind"] == crate::custody::APPOINT {
            snapshot::clear(&self.conn)?;
            self.conn.execute_batch("DELETE FROM replay_jobs; DELETE FROM replay_votes; DELETE FROM replay_claims;")?;
            self.set_meta("started", &false.into())?;
        }
        Ok(())
    }

    fn claim(&mut self, entry: &Value) -> Result<()> {
        let at = float(&entry["at"]);
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
                "refused" if body["refusal"]["code"] == "LOCKED_OUT" && budgets::locked_until(&self.conn, &hotkey, &kind, at)?.is_none() => {
                    self.diverge(entry, SUSPECT, "a miner was refused as locked out while it was not", body.clone(), Value::Null);
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
        Ok(())
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
    /// start, or, with the storage-outage rule on, held while a probe the log holds failed.
    fn forgiveness(&mut self, entry: &Value, kind: &str) -> Result<Option<&'static str>> {
        let input = &entry["input"];
        let basis = &input["forgiving"];
        let expiry = input["expiry"].as_f64();
        let starts: Vec<Value> = self.meta("starts")?.and_then(|s| s.as_array().cloned()).unwrap_or_default();
        let start = starts.last().cloned().unwrap_or(Value::Null);
        if let Some(started_at) = basis["started_at"].as_f64() {
            if !starts.iter().any(|s| s["started_at"].as_f64() == Some(started_at)) {
                self.diverge(entry, FRAUD, "a lapse was decided by a copy the log never saw start", basis["started_at"].clone(), json!(starts));
            }
            let ttl = float(&start["settings"][if kind == EMBED { "embed_claim_ttl" } else { "claim_ttl" }]) + crate::rounds::UPLOAD_GRACE_S;
            if !start.is_null() && basis["claim_ttl"].as_f64() != Some(ttl) {
                self.diverge(entry, FRAUD, "the claim time a restart allows", basis["claim_ttl"].clone(), ttl.into());
            }
            if expiry.is_some_and(|expiry| expiry <= started_at + float(&basis["claim_ttl"])) {
                return Ok(Some("restart"));
            }
        }
        if start["settings"]["storage_outage_forgives"] == true {
            if let (Some(from), Some(to)) = (basis["claimed_at"].as_f64(), expiry) {
                if crate::canary::outage_during(&self.conn, from, to)? {
                    return Ok(Some(crate::canary::OUTAGE_CAUSE));
                }
            }
        } else if basis["claimed_at"].as_f64().is_some() {
            self.diverge(entry, FRAUD, "storage outages were looked up while the rule is off", basis.clone(), Value::Null);
        }
        Ok(None)
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
            if refused["status"] == 422 && matches!(built, Some(Ok(_))) {
                self.diverge(entry, FRAUD, "a feasible report was refused", refused.clone(), Value::Null);
            }
            if refused["status"] == 403 && !validations::is_excluded(&self.conn, &validator)? {
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
        let decision = match text(input, "mode") {
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
                lifecycle::held_to_report(&job, decide(&votes, false, true))
            }
            "unchecked" if kind_of(&job) == EMBED => Decision::last(lifecycle::unchecked_embed_vote(&job), Vec::new()),
            "unchecked" => {
                let share = checks::error_share(&self.conn, text(&job, "miner"), at)?;
                if input["error_share"].as_f64() != Some(share) {
                    self.diverge(entry, FRAUD, "the share of reported errors the miner is paid for", input["error_share"].clone(), share.into());
                }
                Decision::last(lifecycle::unchecked_vote(&job, share), Vec::new())
            }
            _ => Decision::last(lifecycle::unscored_vote(), Vec::new()),
        };
        let vote = decision.vote().clone();
        let result = vote["result"].as_object().cloned().unwrap_or_default();
        let report = validations::build_report(task_id, &job, text(&vote, "validator"), &result, &decision.votes, float(&output["scored_at"]));
        let budget = lifecycle::finalize_accounts(&self.conn, task_id, &job, &decision, Kept { report: &report, publish: None, details: None }, at)?;
        let checked = match budget {
            Some(_) if lifecycle_picked(&job) => lifecycle::check_accounts(&self.conn, &job, &decision, at)?,
            _ => None,
        };
        let ours = json!({
            "verdict": vote["verdict"],
            "reason": vote["result"].get("reason").cloned().unwrap_or(Value::Null),
            "credited": if vote["verdict"] == "pass" { vote["credited"].clone() } else { 0.into() },
            "budget": budget,
            "agreed": decision.agreed,
            "disagreed": decision.disagreed,
            "taken": checked.as_ref().map(|c| c.taken_ids()).unwrap_or_default(),
            "penalty": checked.as_ref().is_some_and(|c| c.penalty.is_some()),
        });
        let mut api = Map::new();
        for name in ["verdict", "reason", "credited", "budget", "agreed", "disagreed", "taken", "penalty"] {
            api.insert(name.into(), output.get(name).cloned().unwrap_or(Value::Null));
        }
        self.compare(entry, "the verdict and its accounts", &Value::Object(api), ours);
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
                };
                registry::is_validator(&neuron).then_some(neuron.hotkey)
            })
            .collect();
        self.compare(entry, "who may vote", &entry["output"]["validators"], json!(validators));
        self.set_meta("validators", &json!(validators))
    }

    fn round(&mut self, entry: &Value, blobs: Blobs) -> Result<()> {
        let output = &entry["output"];
        let sha = output["manifest"]["blob"].as_str().unwrap_or_default();
        let manifest: Option<Vec<Value>> = blobs(sha).filter(|body| proofs::sha256(body) == sha).and_then(|body| serde_json::from_slice(&body).ok());
        let Some(manifest) = manifest else {
            self.diverge(entry, FRAUD, "the round's manifest was not published", output.clone(), Value::Null);
            return Ok(());
        };
        let hash = proofs::manifest_hash(&manifest, int(&output["seed_block"]));
        self.compare(entry, "the round's commitment", &output["manifest_hash"], hash.into());
        for batch in &manifest {
            self.conn.execute(
                "INSERT OR REPLACE INTO replay_batches (batch_id, round_id, urls_hash) VALUES (?, ?, ?)",
                params![text(batch, "batch_id"), text(output, "round_id"), batch["urls_hash"].as_str()],
            )?;
        }
        Ok(())
    }

    fn reveal(&mut self, entry: &Value) -> Result<()> {
        let input = &entry["input"];
        self.note_block(entry, int(&input["seed_block"]), text(input, "block_hash"))?;
        let seed = seed_from_hash(text(input, "block_hash"));
        self.compare(entry, "the round's seed", &entry["output"]["seed"], seed.clone().into());
        let mut statement = self.conn.prepare("SELECT batch_id FROM replay_batches WHERE round_id = ?")?;
        let ids: Vec<String> = statement.query_map([text(input, "round_id")], |row| row.get(0))?.collect::<rusqlite::Result<_>>()?;
        drop(statement);
        if !ids.is_empty() {
            let order = proofs::serve_order(&seed, &ids);
            self.compare(entry, "the round's serve order", &entry["output"]["serve_order"], json!(order));
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
