//! The rules behind a miner's budget, strikes, take-backs and lockouts, and the API's own count of its work.
//!
//! They read only what anyone can recompute: the facts a validator finds in the uploaded file (rows, duplicates, URLs outside
//! the task, rows whose text does not re-extract from their HTML, rows per host), the miner's own signed report, and the
//! round's loadability. Validators report those facts; the API applies the rules to them and never takes a validator's word on
//! whether an upload passed. The live-page comparison is each validator's own business: it reaches the chain through that
//! validator's weights, and the API only reads the miner's on-chain incentive that comes out of them.
//!
//! The same file gives every honest validator the same facts, so facts that differ between two reports mean one of them is
//! wrong. Such an upload is neutral, nothing paid and nothing charged, until anyone checks the public file.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{json, Value};

use crate::budgets::{CEILING, START};
use crate::credit::{overstated, COVERAGE_GATE};
use crate::py;
use crate::verdicts::{UPLOAD_CHANGED, VALIDATOR_FAULT_REASONS};

/// Rows whose text does not re-extract from their HTML that a task may have, as a share of the rows re-extracted.
pub const REEXTRACT_TOLERANCE: f64 = 0.2;
pub const UNREADABLE: &str = "unreadable";
pub const EXTRA_ROWS: &str = "extra_rows";
pub const TEXT_NOT_FROM_HTML: &str = "text_not_from_html";
pub const COVERAGE: &str = "coverage";
pub const REPORTED_ROWS: &str = "reported_rows";
pub const VECTORS_MISMATCH: &str = "vectors_mismatch";
/// Facts about an embed file anyone can read off its bytes, or that storage served another version than the one pinned.
pub const EMBED_FILE_REASONS: [&str; 4] = ["vectors_malformed", "vectors_missing", "wrong_size", UPLOAD_CHANGED];
pub const OK: &str = "ok";
/// An upload paid on the miner's own report, no check having drawn it.
pub const UNCHECKED: &str = "unchecked";
/// Why an upload is neutral: nothing paid, nothing charged.
pub const UNSCORED: &str = "unscored";
pub const FACTS_DISPUTED: &str = "facts_disputed";
pub const BELOW_QUORUM: &str = "below_quorum";
pub const REFERENCES_DISAGREE: &str = "references_disagree";
pub const ONE_REFERENCE: &str = "too_few_references";
pub const ALL_NEUTRAL: &str = "neutral_hosts";

/// What decides an upload's accounts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Credited, and the budget may grow.
    Pass,
    /// The budget halves, a strike, the task's URLs taken back, and everything since the last pass taken back.
    Fail,
    /// Nothing paid and nothing charged.
    Neutral,
}

impl Outcome {
    pub fn name(self) -> &'static str {
        match self {
            Outcome::Pass => "pass",
            Outcome::Fail => "fail",
            Outcome::Neutral => "neutral",
        }
    }
}

/// An upload's accounts: what it is credited, whether it charges or grows the budget, and on which counts.
#[derive(Clone, Debug, PartialEq)]
pub struct Judged {
    pub outcome: Outcome,
    pub reason: String,
    /// Rows (crawl) or characters (embed) credited, as the API counts the miner's work; zero unless it passed.
    pub credited: i64,
    /// The task's URLs outside the round's neutral hosts.
    pub assigned: i64,
    /// The upload's rows for them.
    pub returned: i64,
    pub ramp: bool,
    /// The reports the outcome rests on.
    pub reports: usize,
}

impl Judged {
    fn neutral(reason: &str, assigned: i64, reports: usize) -> Judged {
        Judged { outcome: Outcome::Neutral, reason: reason.into(), credited: 0, assigned, returned: 0, ramp: false, reports }
    }

    pub fn view(&self) -> Value {
        json!({
            "outcome": self.outcome.name(),
            "reason": self.reason,
            "credited": self.credited,
            "assigned": self.assigned,
            "returned": self.returned,
            "ramp": self.ramp,
            "reports": self.reports,
        })
    }
}

/// The knobs of these rules, from the settings; every start entry of the log holds them.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Params {
    pub coverage_gate: f64,
    /// Matching reports an objective fail needs before it charges anything.
    pub objective_quorum: usize,
    /// Validators of distinct coldkeys whose own models must find the same mismatch before an embed upload fails on it.
    pub embed_references: usize,
    /// Validator samples a round needs before its loadability lowers the gate.
    pub loadability_samples: i64,
    /// Checked uploads with rows of a host a round needs before the host can be neutral.
    pub neutral_uploads: i64,
    /// Share of those uploads whose rows of the host are mostly errors past which the host is neutral.
    pub neutral_share: f64,
    /// Share of the incentive its counted work implies at which a miner's budget is no longer capped; 0 turns the cap off.
    pub incentive_full: f64,
    /// Hours of counted work a hotkey has before its incentive caps its budget.
    pub incentive_grace_h: i64,
    /// Each pool's share of the emission, as validators split it.
    pub emission: BTreeMap<String, f64>,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            coverage_gate: COVERAGE_GATE,
            objective_quorum: 1,
            embed_references: 2,
            loadability_samples: 30,
            neutral_uploads: 3,
            neutral_share: 0.5,
            incentive_full: 0.5,
            incentive_grace_h: 24,
            emission: BTreeMap::from([("crawl".to_string(), 0.2), ("embed".to_string(), 0.5)]),
        }
    }
}

/// The host a URL counts under, as `domain_of` gives it in `desearch.canonical`.
pub fn host_of(url: &str) -> String {
    desearch::canonical::domain_of(url).unwrap_or_default()
}

/// What the round's checks showed so far: how much of it validators could load, and the hosts most miners could not.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RoundView {
    pub sampled: i64,
    pub loaded: i64,
    /// Per host: checked uploads with rows of it, and those whose rows of it were mostly errors.
    pub hosts: BTreeMap<String, (i64, i64)>,
    pub loadability: Option<f64>,
    pub neutral: BTreeSet<String>,
}

impl RoundView {
    pub fn decide(mut self, params: &Params) -> RoundView {
        self.loadability = (self.sampled >= params.loadability_samples && self.sampled > 0).then(|| self.loaded as f64 / self.sampled as f64);
        self.neutral = self
            .hosts
            .iter()
            .filter(|(_, (uploads, failed))| *uploads >= params.neutral_uploads && *failed as f64 > params.neutral_share * *uploads as f64)
            .map(|(host, _)| host.clone())
            .collect();
        self
    }

    /// The share of its URLs a task must return: the gate, lowered in proportion when validators could load less of the round.
    pub fn gate(&self, params: &Params) -> f64 {
        params.coverage_gate * self.loadability.unwrap_or(1.0).clamp(0.0, 1.0)
    }

    pub fn view(&self) -> Value {
        json!({
            "sampled": self.sampled,
            "loaded": self.loaded,
            "loadability": self.loadability,
            "hosts": self.hosts.iter().map(|(host, (uploads, failed))| (host.clone(), json!({"uploads": uploads, "mostly_errors": failed}))).collect::<serde_json::Map<_, _>>(),
            "neutral": self.neutral,
        })
    }
}

/// Facts a crawl check found in the file and the task, the same for every honest validator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Facts {
    /// Storage answered the pinned entity tag with another version: the file was written again after its completion.
    pub changed: bool,
    pub readable: bool,
    pub returned: i64,
    pub duplicates: i64,
    pub unassigned: i64,
    pub error_rows: i64,
    pub reextract_mismatch: i64,
    pub integrity_checked: i64,
    /// Rows and error rows per host; older validators do not report them.
    pub hosts: Option<BTreeMap<String, (i64, i64)>>,
}

fn int(value: &Value) -> i64 {
    value.as_i64().or_else(|| value.as_f64().map(|x| x as i64)).unwrap_or(0)
}

/// The facts a crawl vote reports, None when the validator could not judge the file through its own fault.
pub fn facts(vote: &Value) -> Option<Facts> {
    let result = &vote["result"];
    let reason = result["reason"].as_str().unwrap_or_default();
    if VALIDATOR_FAULT_REASONS.contains(&reason) {
        return None;
    }
    if reason == UPLOAD_CHANGED {
        return Some(Facts {
            changed: true,
            readable: true,
            returned: 0,
            duplicates: 0,
            unassigned: 0,
            error_rows: 0,
            reextract_mismatch: 0,
            integrity_checked: 0,
            hosts: None,
        });
    }
    let hosts = result
        .get("hosts")
        .and_then(Value::as_object)
        .map(|hosts| hosts.iter().map(|(host, counts)| (host.clone(), (int(&counts[0]), int(&counts[1])))).collect::<BTreeMap<_, _>>());
    Some(Facts {
        changed: false,
        readable: reason != UNREADABLE,
        returned: int(&result["returned"]),
        duplicates: int(&result["duplicates"]),
        unassigned: int(&result["unassigned"]),
        error_rows: int(&result["error_rows"]),
        reextract_mismatch: int(&result["reextract_mismatch"]),
        integrity_checked: int(&result["integrity_checked"]),
        hosts,
    })
}

/// Two reports agree when every fact both carry is the same; a report without per-host rows agrees on the rest.
fn agree(one: &Facts, other: &Facts) -> bool {
    let strip = |facts: &Facts| Facts { hosts: None, ..facts.clone() };
    strip(one) == strip(other)
        && match (&one.hosts, &other.hosts) {
            (Some(a), Some(b)) => a == b,
            _ => true,
        }
}

fn urls_of(job: &Value) -> Vec<&str> {
    let mut urls: Vec<&str> = job["urls"].as_array().map(|u| u.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
    urls.sort_unstable();
    urls.dedup();
    urls
}

/// The task's URLs on the round's neutral hosts.
pub fn neutral_assigned(job: &Value, round: &RoundView) -> i64 {
    if round.neutral.is_empty() {
        return 0;
    }
    urls_of(job).into_iter().filter(|url| round.neutral.contains(&host_of(url))).count() as i64
}

/// Content and error rows outside neutral hosts, scaled by the task's share of them when the rows per host are unknown.
fn outside_neutral(content: i64, errors: i64, hosts: Option<&BTreeMap<String, (i64, i64)>>, assigned: i64, kept: i64, round: &RoundView) -> (i64, i64) {
    if round.neutral.is_empty() || kept == assigned {
        return (content, errors);
    }
    match hosts {
        Some(hosts) => {
            let (rows, failed) =
                hosts.iter().filter(|(host, _)| round.neutral.contains(*host)).fold((0, 0), |(rows, failed), (_, (r, e))| (rows + r, failed + e));
            ((content - (rows - failed)).max(0), (errors - failed).max(0))
        }
        None if assigned == 0 => (0, 0),
        None => (py::round(content as f64 * kept as f64 / assigned as f64), py::round(errors as f64 * kept as f64 / assigned as f64)),
    }
}

/// Pay as the API counts it: content rows, and error rows at the share of the miner's failures checks reproduced.
fn paid(content: i64, errors: i64, error_share: f64) -> i64 {
    content + py::round(errors as f64 * error_share)
}

/// An upload no validator checked, on the miner's own signed report.
pub fn judge_unchecked(job: &Value, round: &RoundView, error_share: f64, params: &Params) -> Judged {
    let assigned = urls_of(job).len() as i64;
    let reported = &job["reported"];
    let content = int(&reported["ok"]).min(assigned);
    let errors = int(&reported["errors"]).min(assigned - content);
    let kept = assigned - neutral_assigned(job, round);
    if kept == 0 && assigned > 0 {
        return Judged::neutral(ALL_NEUTRAL, 0, 0);
    }
    let (content, errors) = outside_neutral(content, errors, None, assigned, kept, round);
    let returned = content + errors;
    let gate = round.gate(params);
    if (returned as f64) < gate * kept as f64 {
        return Judged { outcome: Outcome::Fail, reason: COVERAGE.into(), credited: 0, assigned: kept, returned, ramp: false, reports: 0 };
    }
    let credited = paid(content, errors, error_share);
    Judged { outcome: Outcome::Pass, reason: UNCHECKED.into(), credited, assigned: kept, returned, ramp: credited as f64 >= gate * kept as f64, reports: 0 }
}

/// Why no vote reported facts: the validators' own reason when they share one.
fn fault_reason(votes: &[Value]) -> &str {
    let reasons: BTreeSet<&str> = votes.iter().filter_map(|vote| vote["result"]["reason"].as_str()).collect();
    match reasons.len() {
        1 => reasons.into_iter().next().unwrap_or(UNSCORED),
        _ => UNSCORED,
    }
}

/// A checked crawl upload, on the facts its validators agree on.
pub fn judge_crawl(job: &Value, votes: &[Value], round: &RoundView, error_share: f64, params: &Params) -> Judged {
    let assigned = urls_of(job).len() as i64;
    let found: Vec<Facts> = votes.iter().filter_map(facts).collect();
    let Some(first) = found.first() else { return Judged::neutral(fault_reason(votes), assigned, 0) };
    if found.iter().any(|other| !agree(first, other)) {
        return Judged::neutral(FACTS_DISPUTED, assigned, found.len());
    }
    let reports = found.len();
    // Read without a pinned version, the key can be rewritten by the bucket's owner or anyone who saw its upload link, not only
    // the miner; read by version, it cannot change at all. Either way a changed upload charges nothing.
    if first.changed {
        return Judged::neutral(UPLOAD_CHANGED, assigned, reports);
    }
    // Per-host rows from whichever report carries them; agreeing reports carry the same.
    let hosts = found.iter().find_map(|f| f.hosts.clone());
    let kept = assigned - neutral_assigned(job, round);
    let returned_all = first.returned.min(assigned);
    let (content, errors) =
        outside_neutral(returned_all - first.error_rows.min(returned_all), first.error_rows.min(returned_all), hosts.as_ref(), assigned, kept, round);
    let returned = content + errors;
    let gate = round.gate(params);
    let reported = &job["reported"];
    let failed = if !first.readable {
        Some(UNREADABLE)
    } else if first.duplicates > 0 || first.unassigned > 0 {
        Some(EXTRA_ROWS)
    } else if first.reextract_mismatch as f64 > REEXTRACT_TOLERANCE * first.integrity_checked as f64 {
        Some(TEXT_NOT_FROM_HTML)
    } else if kept > 0 && (returned as f64) < gate * kept as f64 {
        Some(COVERAGE)
    } else if reported.get("rows").is_some_and(|rows| !rows.is_null())
        && overstated(int(&reported["ok"]), returned_all - first.error_rows.min(returned_all), assigned as usize)
    {
        Some(REPORTED_ROWS)
    } else {
        None
    };
    match failed {
        Some(reason) if reports >= params.objective_quorum.max(1) => {
            Judged { outcome: Outcome::Fail, reason: reason.into(), credited: 0, assigned: kept, returned, ramp: false, reports }
        }
        Some(_) => Judged { returned, ..Judged::neutral(BELOW_QUORUM, kept, reports) },
        None if kept == 0 && assigned > 0 => Judged::neutral(ALL_NEUTRAL, 0, reports),
        None => {
            let credited = paid(content, errors, error_share);
            Judged { outcome: Outcome::Pass, reason: OK.into(), credited, assigned: kept, returned, ramp: credited as f64 >= gate * kept as f64, reports }
        }
    }
}

/// An embed upload no validator checked: every character it was assigned.
pub fn judge_unchecked_embed(job: &Value) -> Judged {
    let credited = int(&job["chars"]);
    Judged { outcome: Outcome::Pass, reason: UNCHECKED.into(), credited, assigned: int(&job["texts"]), returned: int(&job["texts"]), ramp: true, reports: 0 }
}

/// A checked embed upload: a fact of its file fails it, a mismatch only when validators of distinct coldkeys found it with
/// their own models and none found the vectors right.
pub fn judge_embed(job: &Value, votes: &[Value], coldkey_of: &HashMap<String, String>, params: &Params) -> Judged {
    let texts = int(&job["texts"]);
    let judged: Vec<(&str, &str, &str)> = votes
        .iter()
        .filter_map(|vote| {
            let reason = vote["result"]["reason"].as_str().unwrap_or_default();
            let verdict = vote["verdict"].as_str().unwrap_or_default();
            let validator = vote["validator"].as_str().unwrap_or_default();
            (!VALIDATOR_FAULT_REASONS.contains(&reason) && verdict != "void").then_some((validator, verdict, reason))
        })
        .collect();
    if judged.is_empty() {
        return Judged::neutral(fault_reason(votes), texts, 0);
    }
    let file_fails: Vec<&str> =
        judged.iter().filter(|(_, verdict, reason)| *verdict == "fail" && EMBED_FILE_REASONS.contains(reason)).map(|(_, _, reason)| *reason).collect();
    let passes = judged.iter().filter(|(_, verdict, _)| *verdict == "pass").count();
    if let Some(reason) = file_fails.first() {
        if file_fails.len() != judged.len() || file_fails.iter().any(|other| other != reason) {
            return Judged::neutral(FACTS_DISPUTED, texts, judged.len());
        }
        // As for a crawl upload, a changed file is not the miner's doing alone.
        if *reason == UPLOAD_CHANGED {
            return Judged::neutral(UPLOAD_CHANGED, texts, judged.len());
        }
        if judged.len() < params.objective_quorum.max(1) {
            return Judged::neutral(BELOW_QUORUM, texts, judged.len());
        }
        return Judged { outcome: Outcome::Fail, reason: (*reason).into(), credited: 0, assigned: texts, returned: texts, ramp: false, reports: judged.len() };
    }
    let mismatched: BTreeSet<&str> = judged
        .iter()
        .filter(|(_, verdict, reason)| *verdict == "fail" && *reason == VECTORS_MISMATCH)
        .map(|(validator, _, _)| coldkey_of.get(*validator).map(String::as_str).filter(|c| !c.is_empty()).unwrap_or(validator))
        .collect();
    if passes > 0 && !mismatched.is_empty() {
        return Judged::neutral(REFERENCES_DISAGREE, texts, judged.len());
    }
    if !mismatched.is_empty() {
        if mismatched.len() < params.embed_references.max(1) {
            return Judged::neutral(ONE_REFERENCE, texts, judged.len());
        }
        return Judged {
            outcome: Outcome::Fail,
            reason: VECTORS_MISMATCH.into(),
            credited: 0,
            assigned: texts,
            returned: texts,
            ramp: false,
            reports: judged.len(),
        };
    }
    if passes == 0 {
        return Judged::neutral(UNSCORED, texts, judged.len());
    }
    Judged { outcome: Outcome::Pass, reason: OK.into(), credited: int(&job["chars"]), assigned: texts, returned: texts, ramp: true, reports: judged.len() }
}

/// How much of the round validators could load in their samples: the page reached by the validator, whatever the miner got.
pub fn loaded_samples(samples: &[Value]) -> (i64, i64) {
    let loaded = samples.iter().filter(|s| matches!(s["outcome"].as_str(), Some("matched" | "mismatched" | "errors_unconfirmed"))).count();
    (samples.len() as i64, loaded as i64)
}

pub fn create(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS round_loads (
            round_id TEXT PRIMARY KEY,
            sampled  INTEGER NOT NULL DEFAULT 0,
            loaded   INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS round_hosts (
            round_id TEXT NOT NULL,
            host     TEXT NOT NULL,
            uploads  INTEGER NOT NULL DEFAULT 0,
            failed   INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (round_id, host)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS wrong_reports (
            validator TEXT NOT NULL,
            task_id   TEXT NOT NULL,
            why       TEXT NOT NULL,
            at        REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS wrong_reports_validator ON wrong_reports (validator, at);
        ",
    )
}

/// Adds a checked upload's samples and rows per host to its round, before its own accounts are settled on them.
pub fn note_round(conn: &Connection, round_id: &str, votes: &[Value], hosts: Option<&BTreeMap<String, (i64, i64)>>) -> anyhow::Result<()> {
    let (sampled, loaded) = votes.iter().filter(|vote| facts(vote).is_some()).fold((0, 0), |(sampled, loaded), vote| {
        let (s, l) = loaded_samples(vote["result"]["samples"].as_array().map(Vec::as_slice).unwrap_or_default());
        (sampled + s, loaded + l)
    });
    conn.execute(
        "INSERT INTO round_loads VALUES (?, ?, ?) ON CONFLICT (round_id) DO UPDATE SET sampled = sampled + excluded.sampled, loaded = loaded + excluded.loaded",
        params![round_id, sampled, loaded],
    )?;
    for (host, (rows, errors)) in hosts.into_iter().flatten() {
        if *rows == 0 {
            continue;
        }
        conn.execute(
            "INSERT INTO round_hosts VALUES (?, ?, 1, ?) ON CONFLICT (round_id, host) DO UPDATE SET uploads = uploads + 1, failed = failed + excluded.failed",
            params![round_id, host, i64::from(errors * 2 > *rows)],
        )?;
    }
    Ok(())
}

pub fn round_view(conn: &Connection, round_id: &str, params: &Params) -> anyhow::Result<RoundView> {
    let (sampled, loaded) = conn
        .query_row("SELECT sampled, loaded FROM round_loads WHERE round_id = ?", [round_id], |row| Ok((row.get(0)?, row.get(1)?)))
        .optional()?
        .unwrap_or((0, 0));
    let mut statement = conn.prepare("SELECT host, uploads, failed FROM round_hosts WHERE round_id = ?")?;
    let hosts = statement.query_map([round_id], |row| Ok((row.get::<_, String>(0)?, (row.get(1)?, row.get(2)?))))?.collect::<rusqlite::Result<_>>()?;
    Ok(RoundView { sampled, loaded, hosts, ..RoundView::default() }.decide(params))
}

/// Drops the loadability of rounds no longer kept, and wrong reports past any window.
pub fn prune(conn: &Connection, wrong_before: f64) -> anyhow::Result<()> {
    conn.execute("DELETE FROM round_loads WHERE round_id NOT IN (SELECT round_id FROM rounds)", [])?;
    conn.execute("DELETE FROM round_hosts WHERE round_id NOT IN (SELECT round_id FROM rounds)", [])?;
    prune_wrong_reports(conn, wrong_before)
}

/// A report the task could not have produced: the one kind of wrong the API can tell without the file.
pub fn note_wrong_report(conn: &Connection, validator: &str, task_id: &str, why: &str, at: f64) -> anyhow::Result<()> {
    conn.execute("INSERT INTO wrong_reports VALUES (?, ?, ?, ?)", params![validator, task_id, why, at])?;
    Ok(())
}

pub fn wrong_reports(conn: &Connection, validator: &str, since: f64) -> anyhow::Result<i64> {
    Ok(conn.query_row("SELECT COUNT(*) FROM wrong_reports WHERE validator = ? AND at >= ?", params![validator, since], |row| row.get(0))?)
}

pub fn prune_wrong_reports(conn: &Connection, before: f64) -> anyhow::Result<()> {
    conn.execute("DELETE FROM wrong_reports WHERE at < ?", [before])?;
    Ok(())
}

/// The budget a miner's on-chain incentive allows: full once it earns `incentive_full` of what its counted work implies,
/// lowered in proportion below that, never under the starting budget. `ratio` is None when there is nothing to compare.
pub fn incentive_cap(ratio: Option<f64>, params: &Params) -> i64 {
    match ratio {
        Some(ratio) if params.incentive_full > 0.0 => ((CEILING as f64 * (ratio / params.incentive_full).min(1.0)).floor() as i64).clamp(START, CEILING),
        _ => CEILING,
    }
}

/// The incentive a miner's share of each pool implies, as validators split the emission between the pools.
pub fn expected_incentive(shares: &BTreeMap<String, f64>, params: &Params) -> f64 {
    shares.iter().map(|(pool, share)| params.emission.get(pool).copied().unwrap_or(0.0) * share).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(n: usize, hosts: &[&str]) -> Value {
        let urls: Vec<String> = (0..n).map(|i| format!("https://{}/{i}", hosts[i % hosts.len()])).collect();
        json!({"urls": urls, "reported": {"rows": n, "ok": n, "errors": 0}})
    }

    fn vote(validator: &str, result: Value) -> Value {
        let verdict = result.get("verdict").cloned().unwrap_or_else(|| json!("pass"));
        json!({"validator": validator, "verdict": verdict, "result": result})
    }

    fn clean(returned: i64, errors: i64) -> Value {
        json!({"returned": returned, "error_rows": errors, "duplicates": 0, "unassigned": 0, "reextract_mismatch": 0, "integrity_checked": 20, "reason": "ok", "samples": []})
    }

    #[test]
    fn an_objective_fail_charges_whatever_the_live_pages_showed() {
        let params = Params::default();
        let task = job(100, &["a.example"]);
        let mut bad = clean(100, 0);
        bad["duplicates"] = 1.into();
        bad["reason"] = "content_mismatch".into();
        let judged = judge_crawl(&task, &[vote("v1", bad.clone())], &RoundView::default(), 1.0, &params);
        assert_eq!((judged.outcome, judged.reason.as_str()), (Outcome::Fail, EXTRA_ROWS));
        let mut text = clean(100, 0);
        text["reextract_mismatch"] = 5.into();
        assert_eq!(judge_crawl(&task, &[vote("v1", text)], &RoundView::default(), 1.0, &params).reason, TEXT_NOT_FROM_HTML);
        let mut unreadable = clean(0, 0);
        unreadable["reason"] = UNREADABLE.into();
        assert_eq!(judge_crawl(&task, &[vote("v1", unreadable)], &RoundView::default(), 1.0, &params).reason, UNREADABLE);
    }

    #[test]
    fn a_live_page_mismatch_alone_charges_nothing() {
        let params = Params::default();
        let task = job(100, &["a.example"]);
        let mut mismatch = clean(100, 0);
        mismatch["verdict"] = "fail".into();
        mismatch["reason"] = "content_mismatch".into();
        let judged = judge_crawl(&task, &[vote("v1", mismatch.clone()), vote("v2", mismatch)], &RoundView::default(), 1.0, &params);
        assert_eq!((judged.outcome, judged.credited, judged.ramp), (Outcome::Pass, 100, true));
    }

    #[test]
    fn reports_that_disagree_on_a_fact_charge_nothing() {
        let params = Params::default();
        let task = job(100, &["a.example"]);
        let mut lie = clean(100, 0);
        lie["duplicates"] = 3.into();
        let judged = judge_crawl(&task, &[vote("honest", clean(100, 0)), vote("liar", lie)], &RoundView::default(), 1.0, &params);
        assert_eq!((judged.outcome, judged.reason.as_str(), judged.reports), (Outcome::Neutral, FACTS_DISPUTED, 2));
        let mut fault = clean(0, 0);
        fault["reason"] = "unscorable".into();
        let judged = judge_crawl(&task, &[vote("crashed", fault), vote("v2", clean(100, 0))], &RoundView::default(), 1.0, &params);
        assert_eq!((judged.outcome, judged.reports), (Outcome::Pass, 1), "a validator's own crash reports no facts");
        let mut changed = clean(0, 0);
        changed["reason"] = UPLOAD_CHANGED.into();
        let judged = judge_crawl(&task, &[vote("v1", changed.clone())], &RoundView::default(), 1.0, &params);
        assert_eq!((judged.outcome, judged.reason.as_str()), (Outcome::Neutral, UPLOAD_CHANGED), "a rewritten key may not be the miner's doing");
        let judged = judge_crawl(&task, &[vote("v1", changed), vote("v2", clean(100, 0))], &RoundView::default(), 1.0, &params);
        assert_eq!(judged.reason, FACTS_DISPUTED, "one validator read the pinned version, another did not");
    }

    #[test]
    fn a_quorum_holds_back_a_fail_one_validator_found_alone() {
        let params = Params { objective_quorum: 2, ..Params::default() };
        let task = job(10, &["a.example"]);
        let mut bad = clean(10, 0);
        bad["unassigned"] = 1.into();
        assert_eq!(judge_crawl(&task, &[vote("v1", bad.clone())], &RoundView::default(), 1.0, &params).reason, BELOW_QUORUM);
        assert_eq!(judge_crawl(&task, &[vote("v1", bad.clone()), vote("v2", bad)], &RoundView::default(), 1.0, &params).outcome, Outcome::Fail);
    }

    #[test]
    fn the_gate_follows_the_rounds_loadability() {
        let params = Params::default();
        let task = job(100, &["a.example"]);
        let fresh = RoundView { sampled: 10, loaded: 2, ..RoundView::default() }.decide(&params);
        assert_eq!(fresh.loadability, None, "too few samples keep the absolute gate");
        assert_eq!(judge_crawl(&task, &[vote("v1", clean(60, 0))], &fresh, 1.0, &params).reason, COVERAGE);
        let junk = RoundView { sampled: 100, loaded: 60, ..RoundView::default() }.decide(&params);
        assert!((junk.gate(&params) - 0.51).abs() < 1e-9);
        let mut sixty = task.clone();
        sixty["reported"] = json!({"rows": 60, "ok": 60, "errors": 0});
        let judged = judge_crawl(&sixty, &[vote("v1", clean(60, 0))], &junk, 1.0, &params);
        assert_eq!((judged.outcome, judged.ramp), (Outcome::Pass, true), "60 rows clear a gate of 51 and grow the budget");
        let mut unchecked = task.clone();
        unchecked["reported"] = json!({"rows": 55, "ok": 30, "errors": 25});
        assert_eq!(judge_unchecked(&unchecked, &junk, 1.0, &params).outcome, Outcome::Pass);
        assert_eq!(judge_unchecked(&unchecked, &RoundView::default(), 1.0, &params).reason, COVERAGE);
    }

    #[test]
    fn a_host_most_miners_could_not_load_is_neutral_for_everyone() {
        let params = Params::default();
        let round = RoundView {
            hosts: BTreeMap::from([("owner.example".to_string(), (4, 3)), ("fine.example".to_string(), (4, 0)), ("few.example".to_string(), (2, 2))]),
            ..RoundView::default()
        }
        .decide(&params);
        assert_eq!(round.neutral, BTreeSet::from(["owner.example".to_string()]));
        let task = job(100, &["owner.example", "fine.example"]);
        // The miner that could not load the owner's host: 50 error rows there.
        let mut honest = clean(100, 50);
        honest["hosts"] = json!({"owner.example": [50, 50], "fine.example": [50, 0]});
        let mut reported_errors = task.clone();
        reported_errors["reported"] = json!({"rows": 100, "ok": 50, "errors": 50});
        let paid_honest = judge_crawl(&reported_errors, &[vote("v1", honest)], &round, 0.0, &params);
        // The owner's own miner, which loaded it.
        let mut owner = clean(100, 0);
        owner["hosts"] = json!({"owner.example": [50, 0], "fine.example": [50, 0]});
        let paid_owner = judge_crawl(&task, &[vote("v1", owner)], &round, 0.0, &params);
        assert_eq!((paid_honest.assigned, paid_honest.credited, paid_honest.outcome), (50, 50, Outcome::Pass));
        assert_eq!(paid_owner.credited, paid_honest.credited, "neither is paid for the neutral host");
        let mut unchecked = task.clone();
        unchecked["reported"] = json!({"rows": 100, "ok": 100, "errors": 0});
        assert_eq!(judge_unchecked(&unchecked, &round, 1.0, &params).credited, 50, "an unchecked upload is scaled to its task's share outside them");
        let only = job(10, &["owner.example"]);
        assert_eq!(judge_unchecked(&only, &round, 1.0, &params).reason, ALL_NEUTRAL);
    }

    #[test]
    fn an_unchecked_upload_pays_its_report_at_the_confirmed_error_share() {
        let params = Params::default();
        let mut task = job(100, &["a.example"]);
        task["reported"] = json!({"rows": 100, "ok": 80, "errors": 20});
        let judged = judge_unchecked(&task, &RoundView::default(), 0.5, &params);
        assert_eq!((judged.outcome, judged.credited, judged.ramp), (Outcome::Pass, 90, true));
        task["reported"] = json!({"rows": 80, "ok": 60, "errors": 20});
        assert_eq!(judge_unchecked(&task, &RoundView::default(), 0.5, &params).reason, COVERAGE);
    }

    #[test]
    fn an_overstated_report_fails_on_the_file() {
        let params = Params::default();
        let mut task = job(100, &["a.example"]);
        task["reported"] = json!({"rows": 100, "ok": 100, "errors": 0});
        let judged = judge_crawl(&task, &[vote("v1", clean(100, 10))], &RoundView::default(), 1.0, &params);
        assert_eq!((judged.outcome, judged.reason.as_str()), (Outcome::Fail, REPORTED_ROWS));
    }

    fn embed_vote(validator: &str, verdict: &str, reason: &str) -> Value {
        json!({"validator": validator, "verdict": verdict, "result": {"verdict": verdict, "reason": reason}})
    }

    #[test]
    fn an_embed_mismatch_needs_two_independent_references() {
        let params = Params::default();
        let task = json!({"texts": 200, "chars": 5400});
        let coldkeys = HashMap::from([("v1".to_string(), "c1".to_string()), ("v2".to_string(), "c1".to_string()), ("v3".to_string(), "c3".to_string())]);
        let one = judge_embed(&task, &[embed_vote("v1", "fail", VECTORS_MISMATCH)], &coldkeys, &params);
        assert_eq!((one.outcome, one.reason.as_str()), (Outcome::Neutral, ONE_REFERENCE));
        let same_owner = judge_embed(&task, &[embed_vote("v1", "fail", VECTORS_MISMATCH), embed_vote("v2", "fail", VECTORS_MISMATCH)], &coldkeys, &params);
        assert_eq!(same_owner.reason, ONE_REFERENCE, "two hotkeys of one coldkey are one reference");
        let two = judge_embed(&task, &[embed_vote("v1", "fail", VECTORS_MISMATCH), embed_vote("v3", "fail", VECTORS_MISMATCH)], &coldkeys, &params);
        assert_eq!((two.outcome, two.reason.as_str()), (Outcome::Fail, VECTORS_MISMATCH));
        let split = judge_embed(
            &task,
            &[embed_vote("v1", "fail", VECTORS_MISMATCH), embed_vote("v3", "fail", VECTORS_MISMATCH), embed_vote("v9", "pass", "ok")],
            &coldkeys,
            &params,
        );
        assert_eq!(split.reason, REFERENCES_DISAGREE);
        let passed = judge_embed(&task, &[embed_vote("v1", "pass", "ok")], &coldkeys, &params);
        assert_eq!((passed.outcome, passed.credited), (Outcome::Pass, 5400));
        let malformed = judge_embed(&task, &[embed_vote("v1", "fail", "vectors_malformed")], &coldkeys, &params);
        assert_eq!((malformed.outcome, malformed.reason.as_str()), (Outcome::Fail, "vectors_malformed"), "a fact of the file needs no second model");
        let unscorable = judge_embed(&task, &[embed_vote("v1", "void", "unscorable")], &coldkeys, &params);
        assert_eq!(unscorable.outcome, Outcome::Neutral);
    }

    #[test]
    fn the_incentive_caps_a_budget_validators_will_not_pay_for() {
        let params = Params::default();
        assert_eq!(incentive_cap(None, &params), CEILING);
        assert_eq!(incentive_cap(Some(0.6), &params), CEILING);
        assert_eq!(incentive_cap(Some(0.25), &params), 50);
        assert_eq!(incentive_cap(Some(0.0), &params), START);
        assert_eq!(incentive_cap(Some(0.0), &Params { incentive_full: 0.0, ..Params::default() }), CEILING, "0 turns the cap off");
        let shares = BTreeMap::from([("crawl".to_string(), 0.5), ("embed".to_string(), 0.1)]);
        assert!((expected_incentive(&shares, &params) - 0.15).abs() < 1e-12);
    }

    #[test]
    fn samples_say_how_much_validators_could_load() {
        let samples: Vec<Value> = ["matched", "mismatched", "errors_unconfirmed", "errors_confirmed", "unverifiable", "not_fetched"]
            .iter()
            .map(|outcome| json!({"outcome": outcome}))
            .collect();
        assert_eq!(loaded_samples(&samples), (6, 3));
        let conn = Connection::open_in_memory().unwrap();
        create(&conn).unwrap();
        let checked = vote("v1", json!({"reason": "ok", "samples": samples}));
        let hosts = BTreeMap::from([("a.example".to_string(), (10, 8)), ("b.example".to_string(), (10, 1))]);
        for _ in 0..3 {
            note_round(&conn, "r1", std::slice::from_ref(&checked), Some(&hosts)).unwrap();
        }
        let view = round_view(&conn, "r1", &Params { loadability_samples: 18, ..Params::default() }).unwrap();
        assert_eq!((view.sampled, view.loaded, view.loadability), (18, 9, Some(0.5)));
        assert_eq!(view.neutral, BTreeSet::from(["a.example".to_string()]));
    }
}
