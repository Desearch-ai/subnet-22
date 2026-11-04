//! A task's life after the queue: rounds revealed and closed, claims reclaimed, uploads settled by their seed, verdicts finalized with their accounts.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Result};
use bytes::Bytes;
use desearch::canonical::canonicalize;
use desearch::outcomes::DROPPED as OUTCOME_DROPPED;
use desearch::r2::Bucket;
use desearch::time::now;
use futures::stream::{self, StreamExt};
use redis::AsyncCommands;
use rusqlite::Connection;
use serde_json::{json, Map, Value};

use crate::budgets::{self, CRAWL, EMBED, FULL_PENALTY_LOCKOUT_H, HOSTILE_LOCKOUT_H, HOUR, STRIKE_WINDOW_H};
use crate::credit::{self, FAILS_FOR_PENALTY, PENALTY_WINDOW_S};
use crate::manifest::{self, OPEN_LIST_KEY};
use crate::queues::{self, Finalized, OPEN_SCAN};
use crate::roundlog::{self, Receipt};
use crate::rounds::{self, Batch, Round, Url};
use crate::rules::{self, Facts, Judged, Outcome, RoundView};
use crate::sampling;
use crate::settings::Settings;
use crate::state::{Redis, State};
use crate::storage::{delete_quietly, put_json};
use crate::validations::{self, build_report, FinalVerdict};
use crate::verdicts::{decide, Decision, ERROR_OUTCOMES, NO_MAJORITY};
use crate::{checks, hours, outages, outcomes, roundstore};

pub const EMBED_FIELDS: [&str; 7] = ["model", "texts", "chars", "input_key", "input_sha256", "pages", "written_at"];
pub const EMBED_ROUND_INPUTS: isize = 200;
/// Kept past any input's day in storage.
const OPENED_TTL_S: u64 = 2 * 86_400;
pub const FINALIZE_LOCK_S: i64 = 300;
/// Rounds revealed per pass, so a backlog cannot hold the janitor for minutes.
const REVEAL_AT_ONCE: i64 = 64;
const SETTLE_AT_ONCE: usize = 16;
const SLOW_ENQUEUE_S: f64 = 2.0;
/// Malformed-vector reports read per pass, so a flood cannot hold the janitor.
const MALFORMED_AT_ONCE: isize = 100;
/// A check takes about a minute; an upload with less time left would be finalized before its votes land.
pub const CHECKABLE_LEFT_S: f64 = 180.0;
pub const UNCHECKED: &str = "unchecked";
pub const UNSCORED: &str = "unscored";
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

pub fn lock_key(task_id: &str) -> String {
    format!("scoring:{task_id}")
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

/// Two spellings of one page would race for the same key, so each canonical URL is kept once; the round commits to that set,
/// returned with it to be staged until the seed cuts it into batches.
pub fn pack_round(urls: Vec<Url>, target: i64, task_urls: usize, now: f64) -> Result<(Round, Vec<Url>), desearch::canonical::Invalid> {
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
    Ok((rounds::open_set(&unique, target, task_urls, now), unique))
}

/// A committed URL set as staged in Redis until its seed: `[host, url]` pairs.
pub fn staged_set(urls: &[Url]) -> Value {
    urls.iter().map(|u| json!([u.host, u.url])).collect::<Vec<_>>().into()
}

fn set_of(staged: &Value) -> Option<Vec<Url>> {
    staged.as_array()?.iter().map(|pair| Some(Url { host: pair.get(0)?.as_str()?.into(), url: pair.get(1)?.as_str()?.into() })).collect()
}

/// A round committed to its URL set gets its batches from its seed and the set staged for it; false when the set was lost.
async fn cut_round(state: &State, round: &mut Round) -> Result<bool> {
    let (Some(commitment), Some(seed)) = (round.commitment.clone(), round.seed.clone()) else { return Ok(true) };
    let Some(urls) = queues::staged(&state.redis, &round.round_id).await?.as_ref().and_then(set_of) else { return Ok(false) };
    let task_urls = commitment["task_urls"].as_u64().unwrap_or(1) as usize;
    let (round_id, urls_hash) = (round.round_id.clone(), commitment["urls_hash"].as_str().unwrap_or_default().to_string());
    let cutting = seed.clone();
    let (batches, matches) =
        tokio::task::spawn_blocking(move || (rounds::cut(&urls, &cutting, &round_id, task_urls), rounds::set_hash(&urls) == urls_hash)).await?;
    if !matches {
        eprintln!("round={} staged URLs do not match its commitment; it closes without them", round.round_id);
        return Ok(false);
    }
    round.batches = batches;
    rounds::reveal(round, &seed);
    Ok(true)
}

/// The round a bot's URLs make, committed before its seed block exists; returned as SQLite keeps it.
pub async fn open_round(state: &State, urls: Vec<Url>) -> Result<Result<Round, desearch::canonical::Invalid>> {
    let started = Instant::now();
    let target = state.seeds.target_block().await?;
    let blocked = started.elapsed().as_secs_f64();
    let task_urls = state.settings.task_urls;
    let count = urls.len();
    let packing =
        tokio::task::spawn_blocking(move || pack_round(urls, target, task_urls, now()).map(|(round, set)| (queues::pack_job(&staged_set(&set)), round)));
    let (lists, sealed) = match packing.await? {
        Ok(packed) => packed,
        Err(invalid) => return Ok(Err(invalid)),
    };
    let packed = started.elapsed().as_secs_f64();
    save_round(state, &sealed, &lists).await?;
    let took = started.elapsed().as_secs_f64();
    if took > SLOW_ENQUEUE_S {
        eprintln!("enqueueing {count} urls took {took:.1}s (block {blocked:.1}s, pack {:.1}s, save {:.1}s)", packed - blocked, took - packed);
    }
    Ok(Ok(sealed))
}

/// Staged before it is saved, so a saved round always has its URLs.
async fn save_round(state: &State, sealed: &Round, lists: &str) -> Result<()> {
    queues::stage(&state.redis, &sealed.round_id, lists).await?;
    let saved = sealed.clone();
    state.db.run(move |conn| roundstore::save(conn, &saved)).await
}

/// Turns the planner's waiting inputs into one embed round, one batch per input.
pub async fn open_embed_rounds(state: &State) -> Result<Option<Round>> {
    if !state.settings.embed_tasks {
        return Ok(None);
    }
    let mut redis = state.redis.clone();
    let waiting: Vec<String> = redis.lrange(queues::EMBED_INPUTS, 0, EMBED_ROUND_INPUTS - 1).await?;
    if waiting.is_empty() {
        return Ok(None);
    }
    let entries: Vec<Value> = waiting.iter().filter_map(|raw| serde_json::from_str::<Value>(raw).ok()).collect();
    let opened = opened_rounds(state, &entries).await?;
    let model = state.settings.embed_model.clone();
    let mut batches = Vec::new();
    for entry in entries.iter().filter(|entry| !opened.contains(text(entry, "input_key"))) {
        let pages = entry["pages"].as_array().cloned().unwrap_or_default();
        let keep: Vec<Value> =
            pages.iter().map(|p| json!({"page_key": p["page_key"], "content_sha1": p["content_sha1"], "url": p["url"], "host": p["host"]})).collect();
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
        let sealed = round.sealed();
        // Marked before the round is saved, so inputs read again after a crash past the save are not opened twice.
        let mut marks = redis::pipe();
        for batch in &round.batches {
            let input_key = batch.extra.get("input_key").and_then(Value::as_str).unwrap_or_default();
            marks.set_ex(opened_key(input_key), &sealed.round_id, OPENED_TTL_S).ignore();
        }
        let _: () = marks.query_async(&mut redis).await?;
        queues::stage(&state.redis, &sealed.round_id, &queues::pack_job(&round.url_lists())).await?;
        let saved = sealed.clone();
        state.db.run(move |conn| roundstore::save(conn, &saved)).await?;
        opened = Some(sealed);
    }
    // Trimmed only once the round is saved, so a crash re-reads the inputs.
    let _: () = redis.ltrim(queues::EMBED_INPUTS, waiting.len() as isize, -1).await?;
    Ok(opened)
}

fn opened_key(input_key: &str) -> String {
    format!("embed:opened:{input_key}")
}

/// The input keys among `entries` already in a saved round.
async fn opened_rounds(state: &State, entries: &[Value]) -> Result<HashSet<String>> {
    if entries.is_empty() {
        return Ok(HashSet::new());
    }
    let keys: Vec<String> = entries.iter().map(|entry| opened_key(text(entry, "input_key"))).collect();
    let marked: Vec<Option<String>> = redis::cmd("MGET").arg(&keys).query_async(&mut state.redis.clone()).await?;
    let mut saved = HashSet::new();
    for round_id in marked.iter().flatten().collect::<HashSet<_>>() {
        let looked_up = round_id.clone();
        if state.lookups.read(move |conn| roundstore::get(conn, &looked_up)).await?.is_some() {
            saved.insert(round_id.clone());
        }
    }
    Ok(entries
        .iter()
        .zip(&marked)
        .filter(|(_, round_id)| round_id.as_ref().is_some_and(|r| saved.contains(r)))
        .map(|(entry, _)| text(entry, "input_key").to_string())
        .collect())
}

pub async fn reveal_pending(state: &State) -> Result<usize> {
    let current = state.seeds.current_block().await?;
    let pending = state.lookups.read(move |conn| roundstore::unrevealed(conn, current, REVEAL_AT_ONCE)).await?;
    let (mut filled, mut missing_from) = (0, i64::MAX);
    for mut round in pending {
        if round.seed_block >= missing_from {
            continue;
        }
        let seed = match state.seeds.seed_for(round.seed_block).await {
            Ok(Some(seed)) => seed,
            Ok(None) => {
                missing_from = round.seed_block;
                continue;
            }
            Err(error) => {
                eprintln!("round={} waits for its seed: {error:#}", round.round_id);
                missing_from = round.seed_block;
                continue;
            }
        };
        round.seed = Some(seed.clone());
        if !cut_round(state, &mut round).await? {
            eprintln!("round={} lost its URL set before its seed cut it", round.round_id);
        }
        rounds::reveal(&mut round, &seed);
        // Saved first, so a crash cannot queue the round twice.
        let saved = round.clone();
        state.db.run(move |conn| roundstore::mark_revealed(conn, &saved)).await?;
        filled += fill_round(state, &round).await?;
    }
    Ok(filled)
}

/// Queues a revealed round's tasks; it counts as filled only once Redis holds them.
pub async fn fill_round(state: &State, round: &Round) -> Result<usize> {
    let mut round = round.clone();
    if round.commitment.is_some() {
        // Cut again from the staged set after a crash; the seed gives the same batches.
        if !round.has_urls() && !cut_round(state, &mut round).await? {
            eprintln!("round={} lost its URL set before its tasks were queued; it closes without them", round.round_id);
            round.order.clear();
        }
    } else if !round.has_urls() {
        let lists = queues::staged(&state.redis, &round.round_id).await?;
        if !lists.is_some_and(|lists| round.restore_urls(&lists)) {
            eprintln!("round={} lost its URL lists before its tasks were queued; it closes without them", round.round_id);
            round.order.clear();
        }
    }
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
        state.tasks(&round.kind).fill(&state.redis, &round.round_id, &round.order, &payloads, now(), state.settings.release_spread).await?;
    }
    if !round.order.is_empty() {
        let (pages, key, lists) = (state.pages.clone(), round_urls_key(&round.round_id), round.url_lists());
        tokio::spawn(async move {
            let packed = queues::pack_job(&lists);
            if let Err(error) = pages.put(&key, Bytes::from(packed), "text/plain", None).await {
                eprintln!("{key} not kept ({error}); a take-back of its tasks cannot send their URLs again");
            }
        });
    }
    let (round_id, sealed) = (round.round_id.clone(), roundstore::sealed_batches(&round));
    state.db.run(move |conn| roundstore::mark_filled(conn, &round_id, &sealed, now())).await?;
    queues::unstage(&state.redis, &round.round_id).await?;
    state.current.lock().expect("current rounds").insert(round.kind.clone(), round.round_id.clone());
    Ok(round.order.len())
}

/// Rounds revealed before Redis took their tasks get them now, not closed unserved.
pub async fn fill_missing(state: &State) -> Result<Vec<String>> {
    let mut filled = Vec::new();
    for round in state.lookups.read(roundstore::unfilled).await? {
        let open: i64 = state.redis.clone().scard(open_round_key(&round.round_id)).await?;
        if open > 0 {
            let (round_id, sealed) = (round.round_id.clone(), roundstore::sealed_batches(&round));
            state.db.run(move |conn| roundstore::mark_filled(conn, &round_id, &sealed, now())).await?;
            queues::unstage(&state.redis, &round.round_id).await?;
            continue;
        }
        fill_round(state, &round).await?;
        filled.push(round.round_id);
    }
    Ok(filled)
}

pub async fn close_finished(state: &State) -> Result<Vec<String>> {
    let mut closed = Vec::new();
    for round_id in state.lookups.read(roundstore::open_revealed).await? {
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
        let forgiven = forgiven(state, tasks.claim_ttl, expiry).await?;
        if !holder.is_empty() {
            match &forgiven {
                Some(cause) => eprintln!("task={task_id} miner={} lapsed at no cost: its claim was held through a {cause}", short(&holder)),
                None => {
                    let (hotkey, task, lapsed) = (holder.clone(), task_id.clone(), payload.clone());
                    state.db.run(move |conn| lapse(conn, &hotkey, &task, &lapsed, "claim_expired", now)).await?;
                }
            }
            let round_id = Some(text(&payload, "round_id")).filter(|r| !r.is_empty()).map(String::from).unwrap_or_else(|| state.current_round(&kind));
            let cause = forgiven.unwrap_or_else(|| "expired".into());
            state
                .record_one(Receipt {
                    round_id,
                    hotkey: holder.clone(),
                    requested_at: 0.0,
                    outcome: "reclaimed",
                    seq,
                    task_id: Some(task_id.clone()),
                    cause: Some(cause),
                    ..Receipt::default()
                })
                .await?;
        }
        reclaimed.push((task_id, holder));
    }
    Ok(reclaimed)
}

/// Why a claim that ended without an upload costs its miner nothing, if it does: it was held through a restart of the API
/// (`restart`), or alive while storage failed its probe or finality fell behind (`storage_outage`, `finality_outage`).
pub async fn forgiven(state: &State, claim_ttl: f64, expiry: Option<f64>) -> Result<Option<String>> {
    let Some(expiry) = expiry else { return Ok(None) };
    // Claimed before this process started: the miner may have tried to finish while we were down.
    if expiry <= state.started_at() + claim_ttl {
        return Ok(Some("restart".into()));
    }
    // A claim lasts its time plus a block for the one it was read in; one more block keeps a span's edge on the miner's side.
    let from = expiry - claim_ttl - 2.0 * state.clock().await?.block_seconds;
    let cause = state.lookups.read(move |conn| outages::during(conn, from, expiry)).await?;
    Ok(cause.map(|cause| format!("{cause}_outage")))
}

pub fn short(hotkey: &str) -> &str {
    hotkey.get(..10).unwrap_or(hotkey)
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

/// Checks that crashed twice on this upload alone, on every validator that voted, point at the file.
pub fn crashed_on_all(votes: &[Value]) -> bool {
    !votes.is_empty() && votes.iter().all(|vote| truthy(vote["result"].get("crashed")))
}

/// A task that goes no further: its receipt, and its URLs or pages handed back.
pub async fn drop_task(state: &State, task_id: &str, job: &Value, hotkey: &str, cause: &str, seq: i64) -> Result<()> {
    let round_id = text(job, "round_id").to_string();
    state
        .record_one(Receipt {
            round_id: round_id.clone(),
            hotkey: hotkey.into(),
            outcome: "dropped",
            seq,
            task_id: Some(task_id.into()),
            cause: Some(cause.into()),
            ..Receipt::default()
        })
        .await?;
    finish_task(state, &round_id, task_id).await?;
    if kind_of(job) == CRAWL {
        report_outcomes(state, &urls_of(job), OUTCOME_DROPPED, task_id).await;
    } else {
        let dropped = json!({"input_key": job["input_key"], "pages": job["pages"]});
        let _: i64 = state.redis.clone().rpush(queues::EMBED_DROPPED, dropped.to_string()).await?;
    }
    Ok(())
}

/// Why an embed task just claimed goes no further: its input is near the end of its day in storage, or miners handed it back too often.
pub fn spent(settings: &Settings, payload: &Value, handbacks: i64, at: f64) -> Option<&'static str> {
    let written_at = payload.get("written_at").and_then(Value::as_f64);
    if written_at.is_some_and(|written_at| at - written_at > settings.embed_input_max_age) {
        Some("input_expired")
    } else if handbacks >= settings.max_handbacks {
        Some("handed_back")
    } else {
        None
    }
}

/// Drops a task its claimer has not been given yet, at no cost to that miner; false when the claim is no longer theirs.
pub async fn drop_claimed(state: &State, task_id: &str, payload: &Value, hotkey: &str, cause: &str, seq: i64) -> Result<bool> {
    if !state.tasks(kind_of(payload)).discard(&state.redis, task_id, hotkey).await? {
        return Ok(false);
    }
    eprintln!("dropped {task_id} at its claim: {cause}");
    drop_task(state, task_id, payload, "", cause, seq).await?;
    Ok(true)
}

pub async fn requeue(state: &State, task_id: &str, job: &Value, cause: &str) -> Result<()> {
    let kind = kind_of(job).to_string();
    let attempts = number(job, "attempts") + 1;
    let round_id = text(job, "round_id").to_string();
    let miner = text(job, "miner").to_string();
    if attempts >= state.settings.max_attempts {
        let seq = state.next_seq().await?;
        return drop_task(state, task_id, job, &miner, cause, seq).await;
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
    let mut report = build_report(task_id, &lapsed.job, validator, result.as_object().expect("an object"), &[], state.rule_now().await?);
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

/// Uploads whose seed block exists: a drawn share opens for validators, the rest finalize unchecked.
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
    Ok(settled.into_iter().sum())
}

async fn settle_one(state: &State, task_id: &str, job: &Value, seed: &str) -> Result<()> {
    let Some(reason) = pick_reason(state, task_id, job, seed).await? else {
        finalize_unchecked(state, task_id, job, true).await?;
        return Ok(());
    };
    if reason == sampling::RECHECK {
        let (miner, kind) = (text(job, "miner").to_string(), kind_of(job).to_string());
        state.db.run(move |conn| checks::took_recheck(conn, &miner, &kind)).await?;
    }
    let mut picked = job.as_object().cloned().unwrap_or_default();
    picked.insert("picked".into(), reason.into());
    state.validation.open(&state.redis, task_id, &Value::Object(picked), now()).await?;
    Ok(())
}

async fn pick_reason(state: &State, task_id: &str, job: &Value, seed: &str) -> Result<Option<&'static str>> {
    let (kind, miner) = (kind_of(job).to_string(), text(job, "miner").to_string());
    if kind == CRAWL && !truthy(job.get("reported").and_then(|r| r.get("rows"))) {
        return Ok(Some(UNREPORTED));
    }
    let at = state.rule_now().await?;
    let (asked, pool) = (miner.clone(), kind.clone());
    if state.lookups.read(move |conn| budgets::locked_until(conn, &asked, &pool, at)).await?.is_some() {
        return Ok(Some(LOCKED));
    }
    let (asked, pool) = (miner.clone(), kind.clone());
    let (passes, recheck_left) = state.lookups.read(move |conn| Ok((checks::passes(conn, &asked, &pool)?, checks::recheck_left(conn, &asked, &pool)?))).await?;
    let (share, per_hour) = state.settings.checks_for(&kind);
    let share = sampling::budget_share(share, per_hour, sampling::uploads_last_hour(&state.redis, &kind, sampling::ALL, at).await?);
    Ok(sampling::pick_reason(
        sampling::draw(seed, task_id, text(job, "etag")),
        sampling::uploads_last_hour(&state.redis, &kind, &miner, at).await?,
        passes,
        recheck_left,
        share,
    ))
}

/// Embed uploads the publisher found malformed vectors in: their miners' next embed uploads are checked, with no strike or verdict.
pub async fn recheck_malformed(state: &State) -> Result<usize> {
    let mut redis = state.redis.clone();
    let reported: Vec<String> = redis.lrange(queues::EMBED_MALFORMED, 0, MALFORMED_AT_ONCE - 1).await?;
    if reported.is_empty() {
        return Ok(0);
    }
    let found: Vec<Value> = reported.iter().filter_map(|raw| serde_json::from_str(raw).ok()).collect();
    let miners: BTreeSet<String> = found.iter().map(|entry| text(entry, "miner").to_string()).filter(|miner| !miner.is_empty()).collect();
    for entry in &found {
        let miner = text(entry, "miner");
        eprintln!(
            "miner={} embed uploads checked again: task={} had {} malformed vectors",
            miner.get(..10).unwrap_or(miner),
            text(entry, "task_id"),
            entry["malformed"]
        );
    }
    let rechecked = miners.len();
    state
        .db
        .run(move |conn| {
            for miner in &miners {
                checks::start_recheck(conn, miner, EMBED)?;
            }
            Ok(())
        })
        .await?;
    // Trimmed only once the rechecks are saved, so a crash reads the reports again.
    let _: () = redis.ltrim(queues::EMBED_MALFORMED, reported.len() as isize, -1).await?;
    Ok(rechecked)
}

/// An embed upload no validator scored in time: void, at no cost to the miner, and the task goes to another.
pub fn unscored_vote() -> Value {
    json!({
        "validator": "",
        "verdict": "void",
        "credited": 0,
        "result": {"verdict": "void", "reason": UNSCORED, "credited": 0, "samples": []},
    })
}

/// No validator checked it: the miner's own counts, as the rules judged them.
pub fn unchecked_vote(job: &Value, judged: &Judged) -> Value {
    let assigned = assigned(job);
    let reported = &job["reported"];
    let content = number(reported, "ok").min(assigned);
    let errors = number(reported, "errors").min(assigned - content);
    let returned = content + errors;
    let (verdict, reason) = if judged.outcome == Outcome::Fail { ("fail", rules::COVERAGE) } else { ("pass", UNCHECKED) };
    let credited = judged.credited;
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

/// No validator checked it: every character the task was assigned, as a checked pass pays.
pub fn unchecked_embed_vote(job: &Value) -> Value {
    let credited = number(job, "chars");
    json!({
        "validator": "",
        "verdict": "pass",
        "credited": credited,
        "result": {"verdict": "pass", "reason": UNCHECKED, "credited": credited, "returned": number(job, "texts"), "samples": []},
    })
}

/// What an upload's accounts rest on.
#[derive(Clone, Debug)]
pub enum Basis {
    /// No check drew it: the miner's own report.
    Unchecked,
    /// Drawn, but no validator voted by its deadline.
    Unscored,
    /// Every vote it had.
    Checked(Vec<Value>),
}

/// The upload's accounts under the rules, against its round as SQLite holds it now plus this upload's own checks.
pub fn judge(
    conn: &Connection,
    job: &Value,
    basis: &Basis,
    coldkeys: &HashMap<String, String>,
    params: &rules::Params,
    at: f64,
) -> Result<(Judged, RoundView)> {
    let kind = kind_of(job);
    let round_id = text(job, "round_id");
    let mut round = if kind == CRAWL { rules::round_view(conn, round_id, params)? } else { RoundView::default() };
    // Error rows count at the share of the miner's failures checks reproduced, this upload's own checks included.
    let error_share = if kind == CRAWL {
        let ((judged, unconfirmed), (own, own_unconfirmed)) = (checks::error_counts(conn, text(job, "miner"), at)?, error_samples(basis));
        checks::error_rate(judged + own, unconfirmed + own_unconfirmed)
    } else {
        0.0
    };
    let judged = match (basis, kind == EMBED) {
        (Basis::Unchecked, false) => rules::judge_unchecked(job, &round, error_share, params),
        (Basis::Unchecked, true) => rules::judge_unchecked_embed(job),
        (Basis::Unscored, _) => {
            Judged { outcome: Outcome::Neutral, reason: rules::UNSCORED.into(), credited: 0, assigned: assigned(job), returned: 0, ramp: false, reports: 0 }
        }
        (Basis::Checked(votes), true) => rules::judge_embed(job, votes, coldkeys, params),
        (Basis::Checked(votes), false) => {
            // The upload counts toward its own round: its samples, and its rows per host when its reports agree on them.
            let hosts = agreed_hosts(votes);
            let (sampled, loaded) = votes.iter().filter(|vote| rules::facts(vote).is_some()).fold((0, 0), |(s, l), vote| {
                let (vs, vl) = rules::loaded_samples(vote["result"]["samples"].as_array().map(Vec::as_slice).unwrap_or_default());
                (s + vs, l + vl)
            });
            round.sampled += sampled;
            round.loaded += loaded;
            for (host, (rows, errors)) in hosts.iter().flatten() {
                if *rows > 0 {
                    let entry = round.hosts.entry(host.clone()).or_default();
                    entry.0 += 1;
                    entry.1 += i64::from(errors * 2 > *rows);
                }
            }
            round = round.decide(params);
            rules::judge_crawl(job, votes, &round, error_share, params)
        }
    };
    Ok((judged, round))
}

/// Rows per host the reports agree on, None when they disagree on any fact or none carries them.
fn agreed_hosts(votes: &[Value]) -> Option<BTreeMap<String, (i64, i64)>> {
    let found: Vec<rules::Facts> = votes.iter().filter_map(rules::facts).collect();
    let first = found.first()?;
    let same = found.iter().all(|other| Facts { hosts: None, ..other.clone() } == Facts { hosts: None, ..first.clone() });
    let hosts: Vec<&BTreeMap<String, (i64, i64)>> = found.iter().filter_map(|f| f.hosts.as_ref()).collect();
    (same && hosts.windows(2).all(|pair| pair[0] == pair[1])).then(|| hosts.first().map(|h| (*h).clone())).flatten()
}

/// Each validator's coldkey, so references from one owner count once; a hotkey off the metagraph stands for itself.
fn coldkeys(state: &State, votes: &[Value]) -> HashMap<String, String> {
    votes
        .iter()
        .filter_map(|vote| vote["validator"].as_str())
        .map(|validator| (validator.to_string(), state.registry.registered(validator).and_then(|e| e.coldkey).unwrap_or_else(|| validator.to_string())))
        .collect()
}

pub async fn finalize_unchecked(state: &State, task_id: &str, job: &Value, seeding: bool) -> Result<Option<Value>> {
    let (at, params) = (state.rule_now().await?, state.settings.rules());
    let probe = job.clone();
    let (judged, _) = state.lookups.read(move |conn| judge(conn, &probe, &Basis::Unchecked, &HashMap::new(), &params, at)).await?;
    let vote = if kind_of(job) == EMBED { unchecked_embed_vote(job) } else { unchecked_vote(job, &judged) };
    conclude_validation(state, task_id, job, Decision::last(vote, Vec::new()), Basis::Unchecked, seeding).await
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

/// Finalizes once every active validator voted, or at the deadline on the votes it has; with none, a crawl upload on the miner's own counts and an embed upload as void.
pub async fn finalize_task(state: &State, task_id: &str, job: &Value, now: f64) -> Result<Option<Value>> {
    let voters: HashSet<String> = state.validation.voters(&state.redis, task_id).await?.into_iter().collect();
    let mut active: HashSet<String> = state.validation.active_for(&state.redis, kind_of(job), now).await?.into_iter().collect();
    active.extend(voters.iter().cloned());
    let due = now >= job.get("deadline").and_then(Value::as_f64).unwrap_or(0.0);
    if !due && !(!voters.is_empty() && active.is_subset(&voters)) {
        return Ok(None);
    }
    if voters.is_empty() {
        if kind_of(job) != CRAWL {
            return conclude_validation(state, task_id, job, Decision::last(unscored_vote(), Vec::new()), Basis::Unscored, false).await;
        }
        return finalize_unchecked(state, task_id, job, false).await;
    }
    let votes = state.validation.votes(&state.redis, task_id).await?;
    let decision = held_to_report(job, decide(&votes, false, true));
    conclude_validation(state, task_id, job, decision, Basis::Checked(votes), false).await
}

fn overstated(job: &Value, result: &Value) -> bool {
    let reported = &job["reported"];
    if !truthy(reported.get("rows")) {
        return false;
    }
    let counted = number(result, "returned") - number(result, "error_rows");
    credit::overstated(number(reported, "ok"), counted, assigned(job) as usize)
}

/// What is published follows the validators' majority; a pass whose report overstates its file is not published.
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

/// An embed check's sampled texts, each with what it is, its row in the task's input and the URL of its page, as the task page shows them.
fn checked_texts(job: &Value, result: &Map<String, Value>) -> Value {
    let urls: HashMap<&str, &Value> =
        job["pages"].as_array().into_iter().flatten().filter_map(|page| Some((page["page_key"].as_str()?, &page["url"]))).collect();
    let samples = result.get("samples").and_then(Value::as_array).into_iter().flatten();
    samples
        .map(|sample| {
            let text_id = sample["text_id"].as_str().unwrap_or_default();
            let (page, part) = text_id.rsplit_once('#').unwrap_or((text_id, ""));
            let passage = part.strip_prefix('p').and_then(|n| n.parse::<i64>().ok());
            json!({
                "text_id": text_id,
                "kind": if passage.is_some() { "passage" } else { "page" },
                "index": passage.unwrap_or(0),
                "row": sample["index"],
                "url": urls.get(page).copied().cloned().unwrap_or(Value::Null),
                "outcome": sample["outcome"],
                "similarity": sample["similarity"],
            })
        })
        .collect::<Vec<_>>()
        .into()
}

/// The verdict an upload's accounts carry.
pub fn account_verdict(outcome: Outcome) -> &'static str {
    match outcome {
        Outcome::Pass => "pass",
        Outcome::Fail => "fail",
        Outcome::Neutral => "void",
    }
}

/// The report states the accounts as its verdict, reason and credit, and keeps what the validators' majority found beside them.
fn account_report(report: &mut Map<String, Value>, judged: &Judged, published: &str, published_reason: &Value) {
    report.insert("verdict".into(), account_verdict(judged.outcome).into());
    report.insert("reason".into(), judged.reason.clone().into());
    report.insert("credited".into(), judged.credited.into());
    report.insert("publish_verdict".into(), published.into());
    report.insert("publish_reason".into(), published_reason.clone());
    report.insert("accounts".into(), judged.view());
}

/// Published when the validators' majority passed it and its file facts did not fail or conflict.
fn publishes(published: &str, judged: &Judged) -> bool {
    published == "pass" && judged.outcome != Outcome::Fail && judged.reason != rules::FACTS_DISPUTED
}

/// Accounts are finalized once per upload, in one transaction, before Redis lets go; None when storage failed or the upload was closed under it.
pub async fn conclude_validation(state: &State, task_id: &str, job: &Value, decision: Decision, basis: Basis, seeding: bool) -> Result<Option<Value>> {
    let (task, key) = (task_id.to_string(), text(job, "key").to_string());
    if let Some(finalized) = state.lookups.read(move |conn| validations::final_verdict(conn, &task, &key)).await? {
        return finish_finalized(state, task_id, job, finalized, seeding).await;
    }
    let (at, params) = (state.rule_now().await?, state.settings.rules());
    let coldkeys = match &basis {
        Basis::Checked(votes) => coldkeys(state, votes),
        _ => HashMap::new(),
    };
    // Judged once for the report written first; the transaction judges again with whatever of the round settled since.
    let (judged, _) = {
        let (job, basis, coldkeys, params) = (job.clone(), basis.clone(), coldkeys.clone(), params.clone());
        state.lookups.read(move |conn| judge(conn, &job, &basis, &coldkeys, &params, at)).await?
    };
    let vote = decision.vote().clone();
    let result = vote["result"].as_object().cloned().unwrap_or_default();
    let published = text(&vote, "verdict").to_string();
    let published_reason = result.get("reason").cloned().unwrap_or_else(|| "".into());
    let mut report = build_report(task_id, job, text(&vote, "validator"), &result, &decision.votes, at);
    account_report(&mut report, &judged, &published, &published_reason);
    let report_key = text(&Value::Object(report.clone()), "report_key").to_string();
    if let Err(error) = put_json(&state.pages, &report_key, &Value::Object(report.clone()), None).await {
        eprintln!("storage failed while scoring {task_id}: {error}");
        return Ok(None);
    }
    let candidate = (truthy(result.get("matched")) || result.get("reason").and_then(Value::as_str) == Some(UNCHECKED))
        .then(|| publish_job(state, job, &vote, &decision.agreed));
    let details =
        if kind_of(job) == EMBED { validations::pack_details(Some(&checked_texts(job, &result))) } else { validations::pack_details(result.get("urls")) };
    let settled = {
        let (task, job, decision, publish) = (task_id.to_string(), job.clone(), decision.clone(), candidate.clone());
        let (basis, coldkeys, published, published_reason) = (basis.clone(), coldkeys.clone(), published.clone(), published_reason.clone());
        state
            .db
            .run(move |conn| {
                let kept = Kept { report: &report, publish: publish.as_ref(), details: details.as_deref() };
                let given = Given { basis: &basis, coldkeys: &coldkeys, params: &params, published: &published, published_reason: &published_reason };
                finalize_accounts(conn, &task, &job, &decision, given, kept, at)
            })
            .await?
    };
    let Some(settled) = settled else {
        let (task, key) = (task_id.to_string(), text(job, "key").to_string());
        let finalized = state
            .db
            .run(move |conn| validations::final_verdict(conn, &task, &key))
            .await?
            .ok_or_else(|| anyhow!("task={task_id} settled without a verdict"))?;
        return finish_finalized(state, task_id, job, finalized, seeding).await;
    };
    if settled.judged != judged {
        // Another upload of the round settled in between and moved its loadability or neutral hosts.
        if let Err(error) = put_json(&state.pages, &report_key, &Value::Object(settled.report.clone()), None).await {
            eprintln!("could not rewrite the report for {task_id}: {error}");
        }
    }
    let flow = if publishes(&published, &settled.judged) { "pass" } else { "fail" };
    if !close_upload(state, task_id, job, flow, settled.publish.as_ref(), seeding).await? {
        return Ok(None);
    }
    if truthy(job.get("picked")) {
        account_check(state, job, &basis, &settled.judged, &published).await?;
    }
    Ok(Some(json!({
        "task_id": task_id,
        "verdict": account_verdict(settled.judged.outcome),
        "reason": settled.judged.reason,
        "credited": settled.judged.credited,
        "published": flow == "pass",
        "miner_budget": settled.budget,
    })))
}

/// Error samples the checks judged and those the validators could load, summed over every report of the file.
fn error_samples(basis: &Basis) -> (i64, i64) {
    let Basis::Checked(votes) = basis else { return (0, 0) };
    votes.iter().filter(|vote| rules::facts(vote).is_some()).fold((0, 0), |(judged, unconfirmed), vote| {
        let samples = vote["result"]["samples"].as_array().cloned().unwrap_or_default();
        let found = samples.iter().filter(|s| ERROR_OUTCOMES.contains(&text(s, "outcome"))).count() as i64;
        let loaded = samples.iter().filter(|s| text(s, "outcome") == "errors_unconfirmed").count() as i64;
        (judged + found, unconfirmed + loaded)
    })
}

/// A drawn upload's check: an objective pass marks where the hotkey last stood in its kind; an objective fail takes back what
/// passed since then and checks its next uploads. A fail the validators' majority found only on the live pages checks the next
/// uploads too, at no cost: it is evidence for each validator's own weights, not a charge.
async fn account_check(state: &State, job: &Value, basis: &Basis, judged: &Judged, published: &str) -> Result<()> {
    let (miner, kind) = (text(job, "miner").to_string(), kind_of(job).to_string());
    let recheck = published == "fail";
    if judged.outcome == Outcome::Neutral {
        if recheck {
            let (asked, pool) = (miner.clone(), kind.clone());
            state.db.run(move |conn| checks::start_recheck(conn, &asked, &pool)).await?;
        }
        return Ok(());
    }
    let at = state.rule_now().await?;
    let (errors_judged, unconfirmed) = error_samples(basis);
    let passed = judged.outcome == Outcome::Pass;
    let (asked, pool, task_id) = (miner.clone(), kind.clone(), text(job, "task_id").to_string());
    let upload_at = job.get("completed_at").and_then(Value::as_f64).filter(|c| *c != 0.0).unwrap_or(at);
    let since = state
        .db
        .run(move |conn| {
            let since = checks::last_pass_at(conn, &asked, &pool)?;
            checks::record(conn, &asked, &pool, &task_id, passed, upload_at, errors_judged, unconfirmed, at)?;
            if passed && recheck {
                checks::start_recheck(conn, &asked, &pool)?;
            }
            Ok(since)
        })
        .await?;
    if passed {
        return Ok(());
    }
    take_back(state, &miner, &kind, since, "check_failed", false).await?;
    let (asked, pool) = (miner.clone(), kind.clone());
    let fails = state
        .db
        .run(move |conn| {
            checks::start_recheck(conn, &asked, &pool)?;
            checks::fails_in_recent(conn, &asked, &pool)
        })
        .await?;
    if fails >= FAILS_FOR_PENALTY {
        full_penalty(state, &miner, &kind, "repeated_fails").await?;
    }
    Ok(())
}

/// Un-credits and unpublishes a hotkey's passed uploads of one kind completed after `since`: the unchecked ones, or every one.
pub async fn take_back(state: &State, miner: &str, kind: &str, since: f64, reason: &str, checked_too: bool) -> Result<Vec<String>> {
    let at = state.rule_now().await?;
    let (asked, pool, cause) = (miner.to_string(), kind.to_string(), reason.to_string());
    let taken = state
        .db
        .run(move |conn| {
            let taken = validations::withdraw(conn, &asked, &pool, since, checked_too)?;
            let credited: i64 = taken.iter().map(|t| t.credited).sum();
            budgets::credit(conn, &asked, -credited, &pool, at)?;
            if !taken.is_empty() {
                hours::record_take_back(conn, &asked, &pool, &cause, taken.len() as i64, credited, at)?;
            }
            Ok(taken)
        })
        .await?;
    let task_ids: Vec<String> = taken.iter().map(|t| t.task_id.clone()).collect();
    if !task_ids.is_empty() {
        state.publish.withdraw(&state.redis, &task_ids, reason, now()).await?;
        eprintln!("miner={} {} {kind} uploads taken back: {reason}", short(miner), task_ids.len());
        // An embed take-back's pages go back to the planner from the publisher.
        if kind == CRAWL {
            let (pages, storage, redis) = (state.pages.clone(), state.storage.clone(), state.redis.clone());
            tokio::spawn(async move { report_dropped(&pages, &storage, &redis, taken).await });
        }
    }
    Ok(task_ids)
}

/// Where a round's URL lists are kept once its tasks are queued, so a take-back can tell the bot which URLs to send again.
pub fn round_urls_key(round_id: &str) -> String {
    format!("rounds/{round_id}")
}

/// Every URL of the tasks taken back, as dropped in the outcome feed, so the bot sends them again; a round without kept lists is logged and passed over.
pub async fn report_dropped(pages: &Bucket, storage: &Bucket, redis: &Redis, taken: Vec<validations::Taken>) {
    let mut by_round: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for t in taken {
        by_round.entry(t.round_id).or_default().push(t.task_id);
    }
    for (round_id, task_ids) in by_round {
        let lists = match pages.get(&round_urls_key(&round_id)).await {
            Ok(body) => String::from_utf8(body.to_vec()).ok().and_then(|packed| queues::unpack_job(&packed)),
            Err(error) => {
                eprintln!("round={round_id} URL lists unreadable ({error}); {} tasks taken back are not sent again", task_ids.len());
                continue;
            }
        };
        let mut rows = Vec::new();
        for task_id in &task_ids {
            let urls: Vec<String> = lists
                .as_ref()
                .and_then(|lists| lists.get(task_id))
                .and_then(Value::as_array)
                .map(|urls| urls.iter().filter_map(|pair| pair.get(1).and_then(Value::as_str).map(String::from)).collect())
                .unwrap_or_default();
            rows.extend(outcomes::rows_for(&urls, OUTCOME_DROPPED, task_id));
        }
        if let Err(error) = outcomes::write(storage, redis, rows).await {
            eprintln!("round={round_id} dropped outcomes not written: {error:#}");
        }
    }
}

/// The hotkey's last day of credit and published work in one kind's pool, and a lockout from it.
pub async fn full_penalty(state: &State, miner: &str, kind: &str, reason: &str) -> Result<()> {
    let at = state.rule_now().await?;
    let (asked, pool, why) = (miner.to_string(), kind.to_string(), reason.to_string());
    state.db.run(move |conn| budgets::lock_out(conn, &asked, &pool, FULL_PENALTY_LOCKOUT_H, &why, at)).await?;
    take_back(state, miner, kind, at - PENALTY_WINDOW_S, reason, true).await?;
    let (asked, pool) = (miner.to_string(), kind.to_string());
    state.db.run(move |conn| budgets::wipe_credits(conn, &asked, at - PENALTY_WINDOW_S, &pool)).await?;
    eprintln!("miner={} {kind} full penalty: {reason}", short(miner));
    Ok(())
}

pub async fn report_outcomes(state: &State, urls: &[String], outcome: &'static str, task_id: &str) {
    if let Err(error) = outcomes::write(&state.storage, &state.redis, outcomes::rows_for(urls, outcome, task_id)).await {
        eprintln!("could not write {} outcomes for {task_id}: {error:#}", urls.len());
    }
}

/// What a verdict leaves behind: its report, the publish job a pass would have, and the packed per-URL detail of a check.
pub struct Kept<'a> {
    pub report: &'a Map<String, Value>,
    pub publish: Option<&'a Value>,
    pub details: Option<&'a [u8]>,
}

/// What the accounts are judged on, and what the validators' majority found for publishing.
pub struct Given<'a> {
    pub basis: &'a Basis,
    pub coldkeys: &'a HashMap<String, String>,
    pub params: &'a rules::Params,
    pub published: &'a str,
    pub published_reason: &'a Value,
}

/// An upload's accounts as committed.
pub struct Settled {
    pub budget: i64,
    pub judged: Judged,
    pub publish: Option<Value>,
    pub report: Map<String, Value>,
}

/// Every SQLite effect of one verdict, committed together and judged inside the same transaction, so uploads of one round
/// settle in one order; None when the upload was finalized before.
pub fn finalize_accounts(conn: &Connection, task_id: &str, job: &Value, decision: &Decision, given: Given, kept: Kept, now: f64) -> Result<Option<Settled>> {
    let Kept { report, publish, details } = kept;
    let (judged, _) = judge(conn, job, given.basis, given.coldkeys, given.params, now)?;
    let mut report = report.clone();
    account_report(&mut report, &judged, given.published, given.published_reason);
    let publish = publish.filter(|_| publishes(given.published, &judged)).cloned();
    let (kind, miner) = (kind_of(job), text(job, "miner"));
    if !validations::finalize(conn, task_id, text(job, "key"), account_verdict(judged.outcome), judged.credited, publish.as_ref(), now)? {
        return Ok(None);
    }
    if let (Basis::Checked(votes), CRAWL) = (given.basis, kind) {
        rules::note_round(conn, text(job, "round_id"), votes, agreed_hosts(votes).as_ref())?;
    }
    if kind == CRAWL && judged.outcome != Outcome::Neutral {
        budgets::record_coverage(conn, miner, judged.assigned, judged.returned, now)?;
    }
    let budget = match judged.outcome {
        Outcome::Fail => {
            let budget = budgets::penalise(conn, miner, task_id, "verification_failed", kind, now)?.budget;
            if kind == CRAWL {
                budgets::credit(conn, miner, -judged.assigned, kind, now)?;
            }
            budget
        }
        Outcome::Pass if judged.credited != 0 => budgets::reward(conn, miner, task_id, judged.credited, judged.ramp, kind, now)?.budget,
        _ => budgets::get(conn, miner, kind)?.budget,
    };
    // Agreement with the majority is shown on the validators' pages; it no longer costs a validator anything.
    if !decision.agreed.is_empty() || !decision.disagreed.is_empty() {
        validations::record_audit(conn, &decision.agreed, &decision.disagreed)?;
    }
    validations::record(conn, &report, details)?;
    // Each vote is compared with what the majority found, as the votes page shows it.
    let mut majority = report.clone();
    majority.insert("verdict".into(), given.published.into());
    majority.insert("credited".into(), decision.vote()["credited"].clone());
    validations::record_votes(conn, &majority, &decision.votes, &decision.disagreed, decision.vote()["result"]["reason"] != NO_MAJORITY)?;
    if judged.outcome == Outcome::Fail {
        strike(conn, miner, &judged.reason, task_id, kind, now)?;
    }
    if crashed_on_all(&decision.votes) {
        budgets::lock_out(conn, miner, kind, HOSTILE_LOCKOUT_H, "hostile_upload", now)?;
    }
    Ok(Some(Settled { budget, judged, publish, report }))
}

pub fn publish_job(state: &State, job: &Value, vote: &Value, agreed: &[String]) -> Value {
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
    publish.insert("claim_ttl".into(), whole(state.settings.claim_ttl_for(kind)));
    if kind == EMBED {
        for name in ["model", "input_key", "texts", "pages"] {
            publish.insert(name.into(), job[name].clone());
        }
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
}

async fn finish_finalized(state: &State, task_id: &str, job: &Value, finalized: FinalVerdict, seeding: bool) -> Result<Option<Value>> {
    let publish = finalized.publish.map(|mut copy| {
        copy["urls"] = job["urls"].clone();
        copy
    });
    if !close_upload(state, task_id, job, &finalized.verdict, publish.as_ref(), seeding).await? {
        return Ok(None);
    }
    let (miner, kind) = (text(job, "miner").to_string(), kind_of(job).to_string());
    let budget = state.lookups.read(move |conn| budgets::get(conn, &miner, &kind)).await?.budget;
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
