//! A task's life after the queue: rounds revealed and closed, claims reclaimed, uploads settled by their seed, verdicts finalized with their accounts.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Result};
use desearch::canonical::canonicalize;
use desearch::outcomes::DROPPED as OUTCOME_DROPPED;
use desearch::time::{now, utc_day};
use futures::stream::{self, StreamExt};
use redis::AsyncCommands;
use rusqlite::Connection;
use serde_json::{json, Map, Value};

use crate::budgets::{self, COVERAGE_GATE, CRAWL, EMBED, FULL_PENALTY_LOCKOUT_H, HOSTILE_LOCKOUT_H, HOUR, STRIKE_REASONS, STRIKE_WINDOW_H};
use crate::credit::{self, FAILS_FOR_PENALTY, PENALTY_WINDOW_S};
use crate::embeddings::{self, DONE, DROPPED};
use crate::manifest::{self, OPEN_LIST_KEY};
use crate::queues::{self, Finalized, OPEN_SCAN};
use crate::roundlog::{self, Receipt};
use crate::rounds::{self, Batch, Round, Url};
use crate::sampling;
use crate::state::State;
use crate::storage::{delete_quietly, put_json};
use crate::validations::{self, build_report, FinalVerdict};
use crate::verdicts::{decide, Decision, ERROR_OUTCOMES, NO_MAJORITY};
use crate::{checks, outcomes, py, roundstore};

pub const EMBED_FIELDS: [&str; 6] = ["model", "texts", "chars", "input_key", "input_sha256", "pages"];
pub const EMBED_ROUND_INPUTS: isize = 200;
pub const FINALIZE_LOCK_S: i64 = 300;
const SETTLE_AT_ONCE: usize = 16;
const SLOW_ENQUEUE_S: f64 = 2.0;
/// A check takes about a minute; an upload with less time left would be finalized before its votes land.
pub const CHECKABLE_LEFT_S: f64 = 180.0;
pub const UNCHECKED: &str = "unchecked";
pub const REPORTED_ROWS: &str = "reported_rows";
/// Uploads with no counts in their report, or from a locked-out hotkey, are all checked.
pub const UNREPORTED: &str = "unreported";
pub const LOCKED: &str = "locked";

pub fn text<'a>(value: &'a Value, name: &str) -> &'a str {
    value.get(name).and_then(Value::as_str).unwrap_or_default()
}

pub fn kind_of(job: &Value) -> &str {
    job.get("kind").and_then(Value::as_str).unwrap_or(CRAWL)
}

pub fn urls_of(job: &Value) -> Vec<String> {
    job["urls"].as_array().map(|urls| urls.iter().filter_map(Value::as_str).map(String::from).collect()).unwrap_or_default()
}

/// The URLs a task was given, each counted once.
pub fn assigned(job: &Value) -> i64 {
    urls_of(job).into_iter().collect::<HashSet<_>>().len() as i64
}

fn number(value: &Value, name: &str) -> i64 {
    value.get(name).and_then(|v| v.as_i64().or_else(|| v.as_f64().map(|x| x as i64))).unwrap_or(0)
}

fn truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) | Some(Value::Bool(false)) => false,
        Some(Value::Number(n)) => n.as_f64() != Some(0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
        Some(Value::Bool(true)) => true,
    }
}

/// A whole number as an int, the way Python wrote it.
pub fn whole(x: f64) -> Value {
    if x.fract() == 0.0 && x.abs() < 9e15 {
        (x as i64).into()
    } else {
        x.into()
    }
}

pub fn open_round_key(round_id: &str) -> String {
    format!("round:{round_id}:open")
}

pub fn manifest_key(frozen_key: &str) -> String {
    format!("{}.manifest.json", frozen_key.strip_suffix(".parquet").unwrap_or(frozen_key))
}

pub fn lock_key(task_id: &str) -> String {
    format!("scoring:{task_id}")
}

pub fn vectors_key(model: &str, task_id: &str) -> String {
    format!("vectors/model={model}/dt={}/task={task_id}.parquet", utc_day(now()))
}

/// What the miner was given and when it was frozen, signed by the task API.
pub fn signed_manifest(state: &State, job: &Value) -> Value {
    let mut manifest: Map<String, Value> =
        manifest::FIELDS.iter().filter_map(|name| job.get(*name).filter(|v| !v.is_null()).map(|v| (name.to_string(), v.clone()))).collect();
    manifest.insert("signer".into(), state.signer().into());
    let signature = state.sign(&manifest::payload(&manifest));
    manifest.insert("signature".into(), signature.into());
    Value::Object(manifest)
}

/// Writes the open uploads, with their signed manifests, where validators read them.
pub async fn publish_open(state: &State, at: f64) -> Result<bool> {
    let open_ids = state.validation.open_ids(&state.redis, OPEN_SCAN).await?;
    let missing: Vec<String> = {
        let mut known = state.open_manifests.lock().expect("open manifests");
        let open: HashSet<&String> = open_ids.iter().collect();
        known.retain(|task_id, _| open.contains(task_id));
        open_ids.iter().filter(|id| !known.contains_key(*id)).cloned().collect()
    };
    let mut found = Vec::new();
    for task_id in missing {
        if let Some(job) = state.validation.job(&state.redis, &task_id).await? {
            if truthy(job.get("manifest")) {
                found.push((task_id, job["manifest"].clone(), job.get("deadline").and_then(Value::as_f64).unwrap_or(f64::INFINITY)));
            }
        }
    }
    let uploads: Vec<Value> = {
        let mut known = state.open_manifests.lock().expect("open manifests");
        for (task_id, manifest, deadline) in found {
            known.insert(task_id, (manifest, deadline));
        }
        open_ids.iter().filter_map(|id| known.get(id)).filter(|(_, deadline)| deadline - at >= CHECKABLE_LEFT_S).map(|(manifest, _)| manifest.clone()).collect()
    };
    let listed: Vec<String> = uploads.iter().map(|m| text(m, "key").to_string()).collect();
    if state.open_listed.lock().expect("open listing").as_ref() == Some(&listed) {
        return Ok(false);
    }
    let listing = json!({"generated_at": at, "signer": state.signer(), "uploads": uploads});
    if let Err(error) = put_json(&state.storage, OPEN_LIST_KEY, &listing, Some("no-store")).await {
        eprintln!("could not publish the open list: {error}");
        return Ok(false);
    }
    *state.open_listed.lock().expect("open listing") = Some(listed);
    Ok(true)
}

/// Two spellings of one page would race for the same key, so each canonical URL is kept once.
pub fn pack_round(urls: Vec<Url>, target: i64, task_urls: usize, now: f64) -> Result<Round, desearch::canonical::Invalid> {
    let mut at: HashMap<String, usize> = HashMap::new();
    let mut unique: Vec<Url> = Vec::new();
    for url in urls {
        let key = canonicalize(&url.url)?;
        match at.get(&key) {
            Some(&i) => unique[i] = url,
            None => {
                at.insert(key, unique.len());
                unique.push(url);
            }
        }
    }
    Ok(rounds::open_batches(rounds::pack(unique, task_urls), target, "crawl", now))
}

/// The round a bot's URLs make, committed before its seed block exists.
pub async fn open_round(state: &State, urls: Vec<Url>) -> Result<Result<Round, desearch::canonical::Invalid>> {
    let started = Instant::now();
    let target = state.seeds.target_block().await?;
    let blocked = started.elapsed().as_secs_f64();
    let task_urls = state.settings.task_urls;
    let count = urls.len();
    let round = match tokio::task::spawn_blocking(move || pack_round(urls, target, task_urls, now())).await? {
        Ok(round) => round,
        Err(invalid) => return Ok(Err(invalid)),
    };
    let packed = started.elapsed().as_secs_f64();
    let saved = round.clone();
    state.db.run(move |conn| roundstore::save(conn, &saved)).await?;
    let took = started.elapsed().as_secs_f64();
    if took > SLOW_ENQUEUE_S {
        eprintln!("enqueueing {count} urls took {took:.1}s (block {blocked:.1}s, pack {:.1}s, save {:.1}s)", packed - blocked, took - packed);
    }
    Ok(Ok(round))
}

/// Turns the publisher's waiting inputs into one embed round, one batch per input.
pub async fn open_embed_rounds(state: &State) -> Result<Option<Round>> {
    if !state.settings.embed_tasks {
        return Ok(None);
    }
    let mut redis = state.redis.clone();
    let waiting: Vec<String> = redis.lrange(queues::EMBED_INPUTS, 0, EMBED_ROUND_INPUTS - 1).await?;
    if waiting.is_empty() {
        return Ok(None);
    }
    let model = state.settings.embed_model.clone();
    let mut batches = Vec::new();
    for entry in waiting.iter().filter_map(|raw| serde_json::from_str::<Value>(raw).ok()) {
        let pages = entry["pages"].as_array().cloned().unwrap_or_default();
        let checked = pages.clone();
        let wanting = model.clone();
        if state.db.run(move |conn| embeddings::missing(conn, &checked, &wanting)).await?.is_empty() {
            continue;
        }
        let keep: Vec<Value> = pages.iter().map(|p| json!({"page_key": p["page_key"], "content_sha1": p["content_sha1"], "url": p["url"]})).collect();
        let mut extra: Map<String, Value> = EMBED_FIELDS.iter().filter_map(|name| entry.get(*name).map(|v| (name.to_string(), v.clone()))).collect();
        extra.insert("model".into(), model.clone().into());
        extra.insert("pages".into(), keep.into());
        let urls = pages.iter().map(|p| Url { host: text(p, "host").into(), url: text(p, "url").into() }).collect();
        batches.push(Batch::new(rounds::new_id(), urls, extra));
    }
    let mut opened = None;
    if !batches.is_empty() {
        let target = state.seeds.target_block().await?;
        let round = rounds::open_batches(batches, target, EMBED, now());
        let saved = round.clone();
        state
            .db
            .run(move |conn| {
                roundstore::save(conn, &saved)?;
                for batch in &saved.batches {
                    let pages = batch.extra["pages"].as_array().cloned().unwrap_or_default();
                    embeddings::queue(conn, &pages, &model, &batch.batch_id, now())?;
                }
                Ok(())
            })
            .await?;
        opened = Some(round);
    }
    // Trimmed only once the round is saved, so a crash re-reads the inputs.
    let _: () = redis.ltrim(queues::EMBED_INPUTS, waiting.len() as isize, -1).await?;
    Ok(opened)
}

pub async fn reveal_pending(state: &State) -> Result<usize> {
    let current = state.seeds.current_block().await?;
    let pending = state.db.run(move |conn| roundstore::unrevealed(conn, current)).await?;
    let mut filled = 0;
    for mut round in pending {
        let Some(seed) = state.seeds.seed_for(round.seed_block).await? else { continue };
        rounds::reveal(&mut round, &seed);
        // Saved first, so a crash cannot queue the round twice.
        let saved = round.clone();
        state.db.run(move |conn| roundstore::save(conn, &saved)).await?;
        filled += fill_round(state, &round).await?;
    }
    Ok(filled)
}

/// Queues a revealed round's tasks; it counts as filled only once Redis holds them.
pub async fn fill_round(state: &State, round: &Round) -> Result<usize> {
    if !round.order.is_empty() {
        let payloads: Map<String, Value> = round
            .batches
            .iter()
            .map(|batch| {
                let mut payload = batch.extra.clone();
                payload.insert("url_count".into(), batch.urls.len().into());
                payload.insert("urls".into(), batch.urls.iter().map(|u| Value::from(u.url.as_str())).collect::<Vec<_>>().into());
                (batch.batch_id.clone(), Value::Object(payload))
            })
            .collect();
        let _: i64 = state.redis.clone().sadd(open_round_key(&round.round_id), &round.order).await?;
        state.tasks(&round.kind).fill(&state.redis, &round.round_id, &round.order, &payloads).await?;
    }
    let round_id = round.round_id.clone();
    state.db.run(move |conn| roundstore::mark_filled(conn, &round_id, now())).await?;
    state.current.lock().expect("current rounds").insert(round.kind.clone(), round.round_id.clone());
    Ok(round.order.len())
}

/// Rounds revealed before Redis took their tasks get them now, not closed unserved.
pub async fn fill_missing(state: &State) -> Result<Vec<String>> {
    let mut filled = Vec::new();
    for round in state.db.run(roundstore::unfilled).await? {
        let open: i64 = state.redis.clone().scard(open_round_key(&round.round_id)).await?;
        if open > 0 {
            let round_id = round.round_id.clone();
            state.db.run(move |conn| roundstore::mark_filled(conn, &round_id, now())).await?;
            continue;
        }
        fill_round(state, &round).await?;
        filled.push(round.round_id);
    }
    Ok(filled)
}

pub async fn close_finished(state: &State) -> Result<Vec<String>> {
    let mut closed = Vec::new();
    for round_id in state.db.run(roundstore::open_revealed).await? {
        let open: i64 = state.redis.clone().scard(open_round_key(&round_id)).await?;
        if open > 0 {
            continue;
        }
        let closing = round_id.clone();
        state
            .db
            .run(move |conn| {
                roundlog::anchor(conn, &closing, now())?;
                roundstore::close(conn, &closing, now())
            })
            .await?;
        // Later refusals must not land in a log whose root is anchored.
        for current in state.current.lock().expect("current rounds").values_mut() {
            if *current == round_id {
                current.clear();
            }
        }
        closed.push(round_id);
    }
    Ok(closed)
}

pub async fn finish_task(state: &State, round_id: &str, task_id: &str) -> Result<()> {
    let _: i64 = state.redis.clone().srem(open_round_key(round_id), task_id).await?;
    Ok(())
}

pub async fn reclaim_expired(state: &State, now: f64) -> Result<Vec<(String, String)>> {
    let mut reclaimed = Vec::new();
    for task_id in queues::expired_claims(&state.redis, now).await? {
        let payload = state.payload(&task_id).await?.unwrap_or_else(|| json!({}));
        let kind = kind_of(&payload).to_string();
        let tasks = state.tasks(&kind);
        let expiry = queues::claim_expiry(&state.redis, &task_id).await?;
        let Some((holder, seq)) = tasks.reclaim(&state.redis, &task_id, now).await? else { continue };
        // Claimed before this process started: the miner may have tried to finish while we were down.
        let forgiven = expiry.is_some_and(|expiry| expiry <= state.started_at + tasks.claim_ttl);
        if !holder.is_empty() {
            if !forgiven {
                let (hotkey, task, lapsed) = (holder.clone(), task_id.clone(), payload.clone());
                state.db.run(move |conn| lapse(conn, &hotkey, &task, &lapsed, "claim_expired", desearch::time::now())).await?;
            }
            let round_id = Some(text(&payload, "round_id")).filter(|r| !r.is_empty()).map(String::from).unwrap_or_else(|| state.current_round(&kind));
            let cause = if forgiven { "restart" } else { "expired" };
            state
                .record_one(Receipt {
                    round_id,
                    hotkey: holder.clone(),
                    requested_at: 0.0,
                    outcome: "reclaimed",
                    seq,
                    task_id: Some(task_id.clone()),
                    cause: Some(cause.into()),
                    ..Receipt::default()
                })
                .await?;
        }
        reclaimed.push((task_id, holder));
    }
    Ok(reclaimed)
}

/// A claim that ended without an upload: its URLs, the budget and a strike; returns the budget.
pub fn lapse(conn: &Connection, hotkey: &str, task_id: &str, payload: &Value, cause: &str, now: f64) -> Result<i64> {
    let kind = kind_of(payload);
    let assigned = assigned(payload);
    if kind == CRAWL {
        budgets::record_coverage(conn, hotkey, assigned, 0, now)?;
        budgets::credit(conn, hotkey, -assigned, kind, now)?;
    }
    let budget = budgets::penalise(conn, hotkey, task_id, cause, kind, now)?.budget;
    strike(conn, hotkey, cause, task_id, kind, now)?;
    Ok(budget)
}

pub fn strike(conn: &Connection, miner: &str, reason: &str, task_id: &str, kind: &str, now: f64) -> Result<()> {
    let judged = validations::judged_since(conn, miner, now - STRIKE_WINDOW_H * HOUR, kind)?;
    budgets::strike(conn, miner, reason, task_id, judged, kind, now)?;
    Ok(())
}

/// Checks that crashed twice on this upload alone, on most validators, point at the file.
pub fn crashed_on_most(votes: &[Value]) -> bool {
    let crashed = votes.iter().filter(|vote| truthy(vote["result"].get("crashed"))).count();
    crashed * 2 > votes.len()
}

pub async fn requeue(state: &State, task_id: &str, job: &Value, cause: &str) -> Result<()> {
    let kind = kind_of(job).to_string();
    let attempts = number(job, "attempts") + 1;
    let round_id = text(job, "round_id").to_string();
    let miner = text(job, "miner").to_string();
    if attempts >= state.settings.max_attempts {
        let seq = state.next_seq().await?;
        if kind == EMBED {
            let (batch, model) = (task_id.to_string(), text(job, "model").to_string());
            state.db.run(move |conn| embeddings::finalize(conn, &batch, &model, DROPPED, None, now())).await?;
        }
        state
            .record_one(Receipt {
                round_id: round_id.clone(),
                hotkey: miner,
                outcome: "dropped",
                seq,
                task_id: Some(task_id.into()),
                cause: Some(cause.into()),
                ..Receipt::default()
            })
            .await?;
        finish_task(state, &round_id, task_id).await?;
        if kind == CRAWL {
            report_outcomes(state, &urls_of(job), OUTCOME_DROPPED, task_id).await;
        }
        return Ok(());
    }
    let mut payload: Map<String, Value> = EMBED_FIELDS.iter().filter_map(|name| job.get(*name).map(|v| (name.to_string(), v.clone()))).collect();
    payload.insert("kind".into(), kind.clone().into());
    payload.insert("url_count".into(), urls_of(job).len().into());
    payload.insert("urls".into(), job["urls"].clone());
    payload.insert("round_id".into(), round_id.clone().into());
    payload.insert("position".into(), job.get("position").cloned().unwrap_or_else(|| 0.into()));
    payload.insert("rank".into(), job.get("rank").or_else(|| job.get("position")).cloned().unwrap_or_else(|| 0.into()));
    payload.insert("attempts".into(), attempts.into());
    let seq = state.tasks(&kind).restore(&state.redis, task_id, &Value::Object(payload)).await?;
    state
        .record_one(Receipt {
            round_id,
            hotkey: miner,
            outcome: "reclaimed",
            seq,
            task_id: Some(task_id.into()),
            cause: Some(cause.into()),
            ..Receipt::default()
        })
        .await?;
    Ok(())
}

async fn write_report(state: &State, report: &mut Map<String, Value>) {
    let key = report["report_key"].as_str().unwrap_or_default().to_string();
    if let Err(error) = put_json(&state.pages, &key, &Value::Object(report.clone()), None).await {
        eprintln!("could not write the report for {}: {error}", text(&Value::Object(report.clone()), "task_id"));
        report.insert("report_key".into(), "".into());
    }
}

pub async fn void_task(state: &State, task_id: &str, lapsed: Finalized, validator: &str, reason: &str) -> Result<Map<String, Value>> {
    discard_upload(state, &lapsed.job).await;
    requeue(state, task_id, &lapsed.job, "void").await?;
    let result = json!({"verdict": "void", "reason": reason});
    let mut report = build_report(task_id, &lapsed.job, validator, result.as_object().expect("an object"), &[], now());
    write_report(state, &mut report).await;
    let (recorded, votes) = (report.clone(), lapsed.votes);
    state
        .db
        .run(move |conn| {
            validations::record(conn, &recorded, None)?;
            validations::record_votes(conn, &recorded, &votes, &[], false)
        })
        .await?;
    Ok(report)
}

/// Uploads whose seed block exists: a drawn share opens for validators, the rest finalize on the miner's own counts.
pub async fn settle_seeded(state: &Arc<State>) -> Result<usize> {
    let block = match state.seeds.current_block().await {
        Ok(block) => block,
        Err(error) => {
            eprintln!("chain unavailable: {error:#}");
            return Ok(0);
        }
    };
    let ids = state.validation.seeded(&state.redis, block).await?;
    let found: Vec<Result<Option<Value>>> =
        stream::iter(ids.clone()).map(|id| async move { state.validation.job(&state.redis, &id).await }).buffered(SETTLE_AT_ONCE).collect().await;
    let mut jobs = Vec::new();
    for (task_id, job) in ids.into_iter().zip(found) {
        if let Some(job) = job? {
            jobs.push((task_id, job));
        }
    }
    // Uploads frozen together share a seed block, so each block's hash is read once.
    let blocks: BTreeSet<i64> = jobs.iter().map(|(_, job)| number(job, "seed_block")).collect();
    let mut seeds = HashMap::new();
    for block in blocks {
        if let Ok(Some(seed)) = state.seeds.seed_for(block).await {
            seeds.insert(block, seed);
        }
    }
    let settled: Vec<usize> = stream::iter(jobs)
        .map(|(task_id, job)| {
            let seed = seeds.get(&number(&job, "seed_block")).cloned();
            async move {
                let Some(seed) = seed else { return 0 };
                let lock = lock_key(&task_id);
                match state.set_once(&lock, "seeded", FINALIZE_LOCK_S).await {
                    Ok(true) => {}
                    _ => return 0,
                }
                let outcome = settle_one(state, &task_id, &job, &seed).await;
                let _ = state.delete(&lock).await;
                match outcome {
                    Ok(()) => 1,
                    Err(error) => {
                        eprintln!("task={task_id} could not be settled; it is tried again: {error:#}");
                        0
                    }
                }
            }
        })
        .buffer_unordered(SETTLE_AT_ONCE)
        .collect()
        .await;
    let settled = settled.into_iter().sum();
    if settled > 0 {
        publish_open(state, now()).await?;
    }
    Ok(settled)
}

async fn settle_one(state: &State, task_id: &str, job: &Value, seed: &str) -> Result<()> {
    let Some(reason) = pick_reason(state, task_id, job, seed).await? else {
        finalize_unchecked(state, task_id, job, true).await?;
        return Ok(());
    };
    if reason == sampling::RECHECK {
        let miner = text(job, "miner").to_string();
        state.db.run(move |conn| checks::took_recheck(conn, &miner)).await?;
    }
    let mut picked = job.as_object().cloned().unwrap_or_default();
    picked.insert("picked".into(), reason.into());
    state.validation.open(&state.redis, task_id, &Value::Object(picked), now()).await?;
    Ok(())
}

async fn pick_reason(state: &State, task_id: &str, job: &Value, seed: &str) -> Result<Option<&'static str>> {
    let (kind, miner) = (kind_of(job).to_string(), text(job, "miner").to_string());
    if kind == EMBED {
        return Ok(Some(sampling::NEW));
    }
    if !truthy(job.get("reported").and_then(|r| r.get("rows"))) {
        return Ok(Some(UNREPORTED));
    }
    let (asked, pool) = (miner.clone(), kind.clone());
    if state.db.run(move |conn| budgets::locked_until(conn, &asked, &pool, now())).await?.is_some() {
        return Ok(Some(LOCKED));
    }
    let asked = miner.clone();
    let (passes, recheck_left) = state.db.run(move |conn| Ok((checks::passes(conn, &asked)?, checks::recheck_left(conn, &asked)?))).await?;
    let at = now();
    let share =
        sampling::budget_share(state.settings.check_share, state.settings.checks_per_hour, sampling::uploads_last_hour(&state.redis, sampling::ALL, at).await?);
    Ok(sampling::pick_reason(
        sampling::draw(seed, task_id, text(job, "etag")),
        sampling::uploads_last_hour(&state.redis, &miner, at).await?,
        passes,
        recheck_left,
        share,
    ))
}

/// No validator checked it: the miner's own counts, within what it was assigned, errors at its confirmed share.
pub fn unchecked_vote(job: &Value, error_share: f64) -> Value {
    let assigned = assigned(job);
    let reported = &job["reported"];
    let content = number(reported, "ok").min(assigned);
    let errors = number(reported, "errors").min(assigned - content);
    let returned = content + errors;
    let (verdict, reason, credited) = if (returned as f64) < COVERAGE_GATE * assigned as f64 {
        ("fail", "coverage", 0)
    } else {
        ("pass", UNCHECKED, content + py::round(errors as f64 * error_share))
    };
    json!({
        "validator": "",
        "verdict": verdict,
        "credited": credited,
        "result": {
            "verdict": verdict,
            "reason": reason,
            "returned": returned,
            "missing": assigned - returned,
            "error_rows": errors,
            "credited": credited,
            "samples": [],
            "rejected": [],
        },
    })
}

pub async fn finalize_unchecked(state: &State, task_id: &str, job: &Value, seeding: bool) -> Result<Option<Value>> {
    let miner = text(job, "miner").to_string();
    let share = state.db.run(move |conn| checks::error_share(conn, &miner, now())).await?;
    conclude_validation(state, task_id, job, Decision::last(unchecked_vote(job, share), Vec::new()), seeding).await
}

/// Finalizes every open upload that is ready, oldest first.
pub async fn finalize_due(state: &State, now: f64) -> Result<Vec<String>> {
    let mut finalized = Vec::new();
    for task_id in state.validation.open_ids(&state.redis, OPEN_SCAN).await? {
        let Some(job) = state.validation.job(&state.redis, &task_id).await? else { continue };
        let lock = lock_key(&task_id);
        if !state.set_once(&lock, "janitor", FINALIZE_LOCK_S).await? {
            continue;
        }
        let outcome = finalize_task(state, &task_id, &job, now).await;
        state.delete(&lock).await?;
        if outcome?.is_some() {
            finalized.push(task_id);
        }
    }
    Ok(finalized)
}

/// Finalizes once every active validator voted, or at the deadline on the votes it has; with none, on the miner's own counts.
pub async fn finalize_task(state: &State, task_id: &str, job: &Value, now: f64) -> Result<Option<Value>> {
    let voters: HashSet<String> = state.validation.voters(&state.redis, task_id).await?.into_iter().collect();
    let mut active: HashSet<String> = state.validation.active(&state.redis, now).await?.into_iter().collect();
    active.extend(voters.iter().cloned());
    let due = now >= job.get("deadline").and_then(Value::as_f64).unwrap_or(0.0);
    if !due && !(!voters.is_empty() && active.is_subset(&voters)) {
        return Ok(None);
    }
    if voters.is_empty() {
        if kind_of(job) != CRAWL {
            return Ok(None);
        }
        return finalize_unchecked(state, task_id, job, false).await;
    }
    let votes = state.validation.votes(&state.redis, task_id).await?;
    let decision = held_to_report(job, decide(&votes, false, true));
    conclude_validation(state, task_id, job, decision, false).await
}

fn overstated(job: &Value, result: &Value) -> bool {
    let reported = &job["reported"];
    if !truthy(reported.get("rows")) {
        return false;
    }
    let counted = number(result, "returned") - number(result, "error_rows");
    credit::overstated(number(reported, "ok"), counted, assigned(job) as usize)
}

/// Unchecked uploads are paid on the miner's report, so a checked one that overstates it fails.
pub fn held_to_report(job: &Value, decision: Decision) -> Decision {
    let vote = decision.vote();
    if kind_of(job) != CRAWL || vote["verdict"] != "pass" || !overstated(job, &vote["result"]) {
        return decision;
    }
    let mut result = vote["result"].as_object().cloned().unwrap_or_default();
    result.insert("verdict".into(), "fail".into());
    result.insert("reason".into(), REPORTED_ROWS.into());
    result.insert("credited".into(), 0.into());
    let mut failed = vote.as_object().cloned().unwrap_or_default();
    failed.insert("verdict".into(), "fail".into());
    failed.insert("credited".into(), 0.into());
    failed.insert("result".into(), Value::Object(result));
    Decision { vote: Some(Value::Object(failed)), ..decision }
}

/// Accounts are finalized once per upload, in one transaction, before Redis lets go; None when storage failed or the upload was closed under it.
pub async fn conclude_validation(state: &State, task_id: &str, job: &Value, decision: Decision, seeding: bool) -> Result<Option<Value>> {
    let (task, key) = (task_id.to_string(), text(job, "key").to_string());
    if let Some(finalized) = state.db.run(move |conn| validations::final_verdict(conn, &task, &key)).await? {
        return finish_finalized(state, task_id, job, finalized, seeding).await;
    }
    let vote = decision.vote().clone();
    let result = vote["result"].as_object().cloned().unwrap_or_default();
    let verdict = text(&vote, "verdict").to_string();
    let report = build_report(task_id, job, text(&vote, "validator"), &result, &decision.votes, now());
    if let Err(error) = put_json(&state.pages, text(&Value::Object(report.clone()), "report_key"), &Value::Object(report.clone()), None).await {
        eprintln!("storage failed while scoring {task_id}: {error}");
        return Ok(None);
    }
    let publish = (verdict == "pass" && (truthy(result.get("matched")) || result.get("reason").and_then(Value::as_str) == Some(UNCHECKED)))
        .then(|| publish_job(state, task_id, job, &vote, &decision.agreed));
    let settled = {
        let (task, job, decision, report, publish) = (task_id.to_string(), job.clone(), decision.clone(), report.clone(), publish.clone());
        state.db.run(move |conn| finalize_accounts(conn, &task, &job, &decision, &report, publish.as_ref(), now())).await?
    };
    let Some(budget) = settled else {
        let (task, key) = (task_id.to_string(), text(job, "key").to_string());
        let finalized = state
            .db
            .run(move |conn| validations::final_verdict(conn, &task, &key))
            .await?
            .ok_or_else(|| anyhow!("task={task_id} settled without a verdict"))?;
        return finish_finalized(state, task_id, job, finalized, seeding).await;
    };
    for validator in &decision.disagreed {
        let asked = validator.clone();
        if state.db.run(move |conn| validations::is_excluded(conn, &asked)).await? {
            state.validation.leave(&state.redis, validator).await?;
        }
    }
    if !close_upload(state, task_id, job, &verdict, publish.as_ref(), seeding).await? {
        return Ok(None);
    }
    if truthy(job.get("picked")) && kind_of(job) == CRAWL {
        account_check(state, job, &decision).await?;
    }
    Ok(Some(json!({
        "task_id": task_id,
        "verdict": verdict,
        "credited": if verdict == "pass" { vote["credited"].clone() } else { 0.into() },
        "miner_budget": budget,
    })))
}

/// A pass marks where the hotkey last stood; a fail takes back what passed since then and checks its next uploads.
async fn account_check(state: &State, job: &Value, decision: &Decision) -> Result<()> {
    let vote = decision.vote();
    let (verdict, miner) = (text(vote, "verdict").to_string(), text(job, "miner").to_string());
    if verdict != "pass" && verdict != "fail" {
        return Ok(());
    }
    let samples = vote["result"]["samples"].as_array().cloned().unwrap_or_default();
    let judged = samples.iter().filter(|s| ERROR_OUTCOMES.contains(&text(s, "outcome"))).count() as i64;
    let unconfirmed = samples.iter().filter(|s| text(s, "outcome") == "errors_unconfirmed").count() as i64;
    let passed = verdict == "pass";
    let (asked, task_id) = (miner.clone(), text(job, "task_id").to_string());
    let upload_at = job.get("completed_at").and_then(Value::as_f64).filter(|c| *c != 0.0).unwrap_or_else(now);
    let since = state
        .db
        .run(move |conn| {
            let since = checks::last_pass_at(conn, &asked)?;
            checks::record(conn, &asked, &task_id, passed, upload_at, judged, unconfirmed, now())?;
            Ok(since)
        })
        .await?;
    if passed {
        return Ok(());
    }
    take_back(state, &miner, since, "check_failed", false).await?;
    let asked = miner.clone();
    let fails = state
        .db
        .run(move |conn| {
            checks::start_recheck(conn, &asked)?;
            checks::fails_in_recent(conn, &asked)
        })
        .await?;
    if fails >= FAILS_FOR_PENALTY {
        full_penalty(state, &miner, "repeated_fails").await?;
    }
    Ok(())
}

/// Un-credits and unpublishes a hotkey's passed crawl uploads completed after `since`: the unchecked ones, or every one.
pub async fn take_back(state: &State, miner: &str, since: f64, reason: &str, checked_too: bool) -> Result<Vec<String>> {
    let asked = miner.to_string();
    let taken = state
        .db
        .run(move |conn| {
            let taken = validations::withdraw(conn, &asked, since, checked_too)?;
            budgets::credit(conn, &asked, -taken.iter().map(|(_, credited)| credited).sum::<i64>(), CRAWL, now())?;
            Ok(taken)
        })
        .await?;
    let task_ids: Vec<String> = taken.into_iter().map(|(task_id, _)| task_id).collect();
    if !task_ids.is_empty() {
        state.publish.withdraw(&state.redis, &task_ids, reason, now()).await?;
        eprintln!("miner={} {} uploads taken back: {reason}", &miner[..miner.len().min(10)], task_ids.len());
    }
    Ok(task_ids)
}

/// The hotkey's last day of credit and pages, and a lockout.
pub async fn full_penalty(state: &State, miner: &str, reason: &str) -> Result<()> {
    let at = now();
    let (asked, why) = (miner.to_string(), reason.to_string());
    state.db.run(move |conn| budgets::lock_out(conn, &asked, CRAWL, FULL_PENALTY_LOCKOUT_H, &why, at)).await?;
    take_back(state, miner, at - PENALTY_WINDOW_S, reason, true).await?;
    let asked = miner.to_string();
    state.db.run(move |conn| budgets::wipe_credits(conn, &asked, at - PENALTY_WINDOW_S, CRAWL)).await?;
    eprintln!("miner={} full penalty: {reason}", &miner[..miner.len().min(10)]);
    Ok(())
}

pub async fn report_outcomes(state: &State, urls: &[String], outcome: &'static str, task_id: &str) {
    if let Err(error) = outcomes::write(&state.storage, &state.redis, outcomes::rows_for(urls, outcome, task_id)).await {
        eprintln!("could not write {} outcomes for {task_id}: {error:#}", urls.len());
    }
}

/// Every SQLite effect of one verdict, committed together; the miner's budget, or None when the upload was finalized before.
pub fn finalize_accounts(
    conn: &Connection,
    task_id: &str,
    job: &Value,
    decision: &Decision,
    report: &Map<String, Value>,
    publish: Option<&Value>,
    now: f64,
) -> Result<Option<i64>> {
    let vote = decision.vote();
    let result = &vote["result"];
    let (verdict, kind, miner) = (text(vote, "verdict"), kind_of(job), text(job, "miner"));
    let assigned = assigned(job);
    let credited = if verdict == "pass" { number(vote, "credited") } else { 0 };
    let reason = text(result, "reason");
    if !validations::finalize(conn, task_id, text(job, "key"), verdict, credited, publish, now)? {
        return Ok(None);
    }
    if kind == CRAWL && verdict != "void" {
        budgets::record_coverage(conn, miner, assigned, number(result, "returned"), now)?;
    }
    let budget = if verdict == "fail" {
        let budget = budgets::penalise(conn, miner, task_id, "verification_failed", kind, now)?.budget;
        if kind == CRAWL && STRIKE_REASONS.contains(&reason) {
            budgets::credit(conn, miner, -assigned, kind, now)?;
        }
        budget
    } else if credited != 0 {
        // An embed pass is all or nothing, so every one grows the budget.
        let ramp = kind == EMBED || credited as f64 >= COVERAGE_GATE * assigned as f64;
        budgets::reward(conn, miner, task_id, credited, ramp, kind, now)?.budget
    } else {
        budgets::get(conn, miner, kind)?.budget
    };
    if let (Some(publish), true) = (publish, kind == EMBED) {
        embeddings::finalize(conn, task_id, text(job, "model"), DONE, publish.get("vectors_key").and_then(Value::as_str), now)?;
    }
    if !decision.agreed.is_empty() || !decision.disagreed.is_empty() {
        validations::record_audit(conn, &decision.agreed, &decision.disagreed)?;
    }
    validations::record(conn, report, result.get("urls"))?;
    validations::record_votes(conn, report, &decision.votes, &decision.disagreed, reason != NO_MAJORITY)?;
    if verdict == "fail" && STRIKE_REASONS.contains(&reason) {
        strike(conn, miner, reason, task_id, kind, now)?;
    }
    if crashed_on_most(&decision.votes) {
        budgets::lock_out(conn, miner, kind, HOSTILE_LOCKOUT_H, "hostile_upload", now)?;
    }
    Ok(Some(budget))
}

pub fn publish_job(state: &State, task_id: &str, job: &Value, vote: &Value, agreed: &[String]) -> Value {
    let kind = kind_of(job);
    let mut publish: Map<String, Value> =
        ["task_id", "round_id", "miner", "key", "etag"].iter().map(|name| (name.to_string(), job.get(*name).cloned().unwrap_or_else(|| "".into()))).collect();
    let validator = text(vote, "validator");
    let mut validators: BTreeSet<String> = agreed.iter().cloned().collect();
    if !validator.is_empty() {
        validators.insert(validator.into());
    }
    publish.insert("kind".into(), kind.into());
    publish.insert("validator".into(), vote["validator"].clone());
    publish.insert("validators".into(), validators.into_iter().collect::<Vec<_>>().into());
    publish.insert("urls".into(), job["urls"].clone());
    publish.insert("completed_at".into(), job["completed_at"].clone());
    publish.insert("claim_ttl".into(), whole(state.settings.claim_ttl));
    if kind == EMBED {
        for name in ["model", "input_key", "pages", "texts"] {
            publish.insert(name.into(), job[name].clone());
        }
        publish.insert("vectors_key".into(), vectors_key(text(job, "model"), task_id).into());
    } else {
        publish.insert("skip".into(), rejected_urls(&vote["result"]).into());
    }
    Value::Object(publish)
}

/// Rows a check failed stay out of the corpus even though the task passed.
pub fn rejected_urls(result: &Value) -> Vec<String> {
    let mut rejected: BTreeSet<String> =
        result["samples"].as_array().into_iter().flatten().filter(|s| text(s, "outcome") == "mismatched").map(|s| text(s, "url").to_string()).collect();
    rejected.extend(result["rejected"].as_array().into_iter().flatten().filter_map(Value::as_str).map(String::from));
    rejected.into_iter().collect()
}

/// Closes a finalized upload in Redis and moves the task on; false when it was closed before its verdict was recorded.
async fn close_upload(state: &State, task_id: &str, job: &Value, verdict: &str, publish: Option<&Value>, seeding: bool) -> Result<bool> {
    if state.validation.finalize(&state.redis, task_id, publish, seeding, now()).await?.is_none() {
        eprintln!("task={task_id} was closed before its final_verdict was recorded");
        return Ok(false);
    }
    if verdict == "pass" {
        finish_task(state, text(job, "round_id"), task_id).await?;
    } else {
        discard_upload(state, job).await;
        requeue(state, task_id, job, verdict).await?;
    }
    Ok(true)
}

/// Only a passed upload is published, and a kept file could be resubmitted by the next miner.
pub async fn discard_upload(state: &State, job: &Value) {
    let key = text(job, "key");
    delete_quietly(&state.storage, key).await;
    delete_quietly(&state.storage, &manifest_key(key)).await;
}

async fn finish_finalized(state: &State, task_id: &str, job: &Value, finalized: FinalVerdict, seeding: bool) -> Result<Option<Value>> {
    if !close_upload(state, task_id, job, &finalized.verdict, finalized.publish.as_ref(), seeding).await? {
        return Ok(None);
    }
    let (miner, kind) = (text(job, "miner").to_string(), kind_of(job).to_string());
    let budget = state.db.run(move |conn| budgets::get(conn, &miner, &kind)).await?.budget;
    Ok(Some(json!({"task_id": task_id, "verdict": finalized.verdict, "credited": finalized.credited, "miner_budget": budget})))
}

pub async fn return_expired_publishes(state: &State) -> Result<Vec<String>> {
    let mut returned = Vec::new();
    for task_id in state.publish.expired(&state.redis, now()).await? {
        let outcome = state.publish.give_back(&state.redis, &task_id).await?;
        if outcome < 0 {
            eprintln!("task={task_id} failed to publish {} times; set aside", state.publish.max_tries);
        } else if outcome > 0 {
            returned.push(task_id);
        }
    }
    Ok(returned)
}
