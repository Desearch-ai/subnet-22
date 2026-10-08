//! The public log pages: miners, validators, votes, events and tasks, read on a connection of their own and cached for a few seconds.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;
use axum::extract::{Path, Query as QueryParams, State as Shared};
use axum::Json;
use desearch::time::now;
use redis::AsyncCommands;
use rusqlite::Connection;
use serde_json::{json, Map, Value};

use crate::budgets::{self, CRAWL, HOUR, SHARE_WINDOW_H};
use crate::http::{Answer, ApiError, Query};
use crate::lifecycle::{kind_of, text};
use crate::models::Invalid;
use crate::py::round_to;
use crate::queues::{CLAIMS, KINDS};
use crate::roundlog;
use crate::state::State;
use crate::validations::{self, TaskFilter, VoteFilter, VERDICTS};

const CACHE_FOR: Duration = Duration::from_secs(5);
const CACHE_ENTRIES: usize = 256;
const PAGE: i64 = 50;
const MAX_PAGE: i64 = 100;
const MAX_WINDOW_H: i64 = 168;
const LIVE_ROWS: isize = 50;
const BUCKET_MINUTES: [i64; 3] = [5, 15, 60];
const MAX_BUCKETS: i64 = 168;
const ID_CHARS: usize = 64;
pub const PATHS: [&str; 8] = ["/v1/overview", "/v1/live", "/v1/stats", "/v1/tasks", "/v1/votes", "/v1/events", "/v1/miners", "/v1/validators"];
const MINER_TOTALS: [&str; 7] = ["tasks", "pass", "fail", "void", "returned", "missing", "credited"];
const VALIDATOR_TOTALS: [&str; 7] = ["votes", "pass", "fail", "void", "agreed", "disagreed", "decided"];

/// Pages computed in the last few seconds, so a busy dashboard reads SQLite once.
#[derive(Default)]
pub struct Cache {
    entries: Mutex<HashMap<String, (Instant, Value)>>,
}

impl Cache {
    pub async fn get_or<F: Future<Output = Result<Value, ApiError>>>(&self, key: String, make: impl FnOnce() -> F) -> Result<Value, ApiError> {
        if let Some((until, value)) = self.entries.lock().expect("cache").get(&key) {
            if *until > Instant::now() {
                return Ok(value.clone());
            }
        }
        let value = make().await?;
        let mut entries = self.entries.lock().expect("cache");
        if entries.len() >= CACHE_ENTRIES {
            entries.clear();
        }
        entries.insert(key, (Instant::now() + CACHE_FOR, value.clone()));
        Ok(value)
    }
}

fn invalid(name: &str, kind: &str, msg: &str) -> ApiError {
    Invalid::single(kind, &["query", name], msg).into()
}

pub fn int_param(query: &Query, name: &str, default: i64, ge: i64, le: i64) -> Result<i64, ApiError> {
    let Some(raw) = query.get(name) else { return Ok(default) };
    let value: i64 = raw.trim().parse().map_err(|_| invalid(name, "int_parsing", "Input should be a valid integer, unable to parse string as an integer"))?;
    if value < ge {
        return Err(invalid(name, "greater_than_equal", &format!("Input should be greater than or equal to {ge}")));
    }
    if value > le {
        return Err(invalid(name, "less_than_equal", &format!("Input should be less than or equal to {le}")));
    }
    Ok(value)
}

fn optional_int(query: &Query, name: &str) -> Result<Option<i64>, ApiError> {
    query
        .get(name)
        .map(|raw| raw.trim().parse().map_err(|_| invalid(name, "int_parsing", "Input should be a valid integer, unable to parse string as an integer")))
        .transpose()
}

pub fn float_param(query: &Query, name: &str) -> Result<Option<f64>, ApiError> {
    query
        .get(name)
        .map(|raw| raw.trim().parse().map_err(|_| invalid(name, "float_parsing", "Input should be a valid number, unable to parse string as a number")))
        .transpose()
}

fn id_param(query: &Query, name: &str) -> Result<Option<String>, ApiError> {
    match query.get(name) {
        Some(value) if value.chars().count() > ID_CHARS => Err(invalid(name, "string_too_long", &format!("String should have at most {ID_CHARS} characters"))),
        other => Ok(other.cloned()),
    }
}

fn literal_param(query: &Query, name: &str, allowed: &[&str]) -> Result<Option<String>, ApiError> {
    match query.get(name) {
        Some(value) if !allowed.contains(&value.as_str()) => {
            let quoted: Vec<String> = allowed.iter().map(|a| format!("'{a}'")).collect();
            Err(invalid(name, "literal_error", &format!("Input should be {}", quoted.join(" or "))))
        }
        other => Ok(other.cloned()),
    }
}

fn bool_param(query: &Query, name: &str) -> Result<Option<bool>, ApiError> {
    let Some(raw) = query.get(name) else { return Ok(None) };
    match raw.to_lowercase().as_str() {
        "1" | "true" | "t" | "yes" | "y" | "on" => Ok(Some(true)),
        "0" | "false" | "f" | "no" | "n" | "off" => Ok(Some(false)),
        _ => Err(invalid(name, "bool_parsing", "Input should be a valid boolean, unable to interpret input")),
    }
}

fn path_id(id: &str) -> Result<(), ApiError> {
    if id.chars().count() > ID_CHARS {
        return Err(Invalid::single("string_too_long", &["path", "id"], &format!("String should have at most {ID_CHARS} characters")).into());
    }
    Ok(())
}

fn visible(state: &State) -> f64 {
    now() - state.settings.ledger_delay
}

fn uid_of(state: &State, hotkey: Option<&str>) -> Value {
    hotkey.and_then(|h| state.registry.registered(h)).and_then(|e| e.uid).map_or(Value::Null, Value::from)
}

/// The row with a uid beside each hotkey it names; null for one off the metagraph.
fn with_uids(state: &State, mut row: Map<String, Value>) -> Value {
    for name in ["miner", "validator"] {
        if let Some(hotkey) = row.get(name).cloned() {
            row.insert(format!("{name}_uid"), uid_of(state, hotkey.as_str()));
        }
    }
    Value::Object(row)
}

fn with_keys(state: &State, mut row: Map<String, Value>, hotkey: &str) -> Value {
    let entry = state.registry.registered(hotkey);
    row.insert("uid".into(), entry.as_ref().and_then(|e| e.uid).map_or(Value::Null, Value::from));
    row.insert("coldkey".into(), entry.and_then(|e| e.coldkey).map_or(Value::Null, Value::from));
    Value::Object(row)
}

fn number(value: &Value) -> f64 {
    value.as_f64().unwrap_or(0.0)
}

fn miner_rows_at(conn: &Connection, since: f64, until: f64) -> Result<Vec<Map<String, Value>>> {
    let totals = validations::miner_totals(conn, since, until)?;
    let budgets: BTreeMap<String, budgets::MinerBudget> = budgets::all(conn, CRAWL)?.into_iter().map(|b| (b.hotkey.clone(), b)).collect();
    let locked = budgets::lockouts(conn, CRAWL, now())?;
    let coverage = budgets::coverage_report(conn, SHARE_WINDOW_H, until)?;
    let shares = budgets::shares(conn, SHARE_WINDOW_H, until)?.remove(CRAWL).unwrap_or_default();
    let hotkeys: BTreeSet<&String> = totals.keys().chain(budgets.keys()).collect();
    let mut rows = Vec::with_capacity(hotkeys.len());
    for hotkey in hotkeys {
        let budget = match budgets.get(hotkey) {
            Some(budget) => budget.clone(),
            None => budgets::get(conn, hotkey, CRAWL)?,
        };
        let mut row = Map::new();
        row.insert("hotkey".into(), hotkey.clone().into());
        row.insert("budget".into(), budget.budget.into());
        row.insert("locked_until".into(), locked.get(hotkey).copied().map_or(Value::Null, Value::from));
        row.insert("verified".into(), budget.verified.into());
        for name in MINER_TOTALS {
            row.insert(name.into(), 0.into());
        }
        row.insert("last_scored_at".into(), Value::Null);
        if let Some(Value::Object(total)) = totals.get(hotkey) {
            row.extend(total.clone());
        }
        row.insert("coverage".into(), coverage.get(hotkey).and_then(|c| c.get("coverage")).cloned().unwrap_or(Value::Null));
        row.insert("share".into(), shares.get(hotkey).copied().unwrap_or(0.0).into());
        rows.push(row);
    }
    rows.sort_by(|a, b| {
        (-number(&a["share"]), -number(&a["credited"])).partial_cmp(&(-number(&b["share"]), -number(&b["credited"]))).unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(rows)
}

fn validator_row(hotkey: &str, totals: Option<&Value>, standing: Option<&Value>, last_seen: &HashMap<String, f64>) -> Map<String, Value> {
    let mut counts: Map<String, Value> = VALIDATOR_TOTALS.iter().map(|name| (name.to_string(), Value::from(0))).collect();
    counts.insert("last_vote_at".into(), Value::Null);
    if let Some(Value::Object(totals)) = totals {
        counts.extend(totals.clone());
    }
    let judged = number(&counts["agreed"]) + number(&counts["disagreed"]);
    let mut row = Map::new();
    row.insert("hotkey".into(), hotkey.into());
    row.insert("active".into(), last_seen.contains_key(hotkey).into());
    row.insert("last_seen".into(), last_seen.get(hotkey).copied().map_or(Value::Null, Value::from));
    row.extend(counts.clone());
    row.insert("agreement".into(), if judged > 0.0 { round_to(number(&counts["agreed"]) / judged, 4).into() } else { Value::Null });
    row.insert("audits".into(), 0.into());
    row.insert("disagreements".into(), 0.into());
    row.insert("excluded".into(), false.into());
    if let Some(Value::Object(standing)) = standing {
        row.extend(standing.clone());
    }
    row
}

fn validator_rows_at(conn: &Connection, since: f64, until: f64, last_seen: &HashMap<String, f64>) -> Result<Vec<Map<String, Value>>> {
    let totals = validations::validator_totals(conn, since, until)?;
    let standing = validations::audit_standing(conn)?;
    let hotkeys: BTreeSet<&String> = totals.keys().chain(standing.keys()).chain(last_seen.keys()).collect();
    let mut rows: Vec<Map<String, Value>> = hotkeys.into_iter().map(|h| validator_row(h, totals.get(h), standing.get(h), last_seen)).collect();
    rows.sort_by(|a, b| {
        (-number(&a["votes"]), text(&Value::Object(a.clone()), "hotkey").to_string())
            .partial_cmp(&(-number(&b["votes"]), text(&Value::Object(b.clone()), "hotkey").to_string()))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(rows)
}

async fn miner_rows(state: &Arc<State>, hours: i64) -> Result<Value, ApiError> {
    let until = visible(state);
    let rows = state.reader.read(move |conn| miner_rows_at(conn, until - hours as f64 * HOUR, until)).await?;
    let mut out = Vec::with_capacity(rows.len());
    for mut row in rows {
        let hotkey = text(&Value::Object(row.clone()), "hotkey").to_string();
        row.insert("uid".into(), uid_of(state, Some(&hotkey)));
        row.insert("in_flight".into(), state.crawl.in_flight(&state.redis, &hotkey).await?.into());
        row.insert("waiting".into(), state.crawl.waiting(&state.redis, &hotkey).await?.into());
        out.push(Value::Object(row));
    }
    Ok(out.into())
}

async fn validator_rows(state: &Arc<State>, hours: i64) -> Result<Value, ApiError> {
    let until = visible(state);
    let last_seen: HashMap<String, f64> = state.validation.last_seen(&state.redis, now()).await?.into_iter().collect();
    let rows = state.reader.read(move |conn| validator_rows_at(conn, until - hours as f64 * HOUR, until, &last_seen)).await?;
    Ok(rows
        .into_iter()
        .map(|mut row| {
            let uid = uid_of(state, row.get("hotkey").and_then(Value::as_str));
            row.insert("uid".into(), uid);
            Value::Object(row)
        })
        .collect::<Vec<_>>()
        .into())
}

async fn cached_miners(state: &Arc<State>, hours: i64) -> Result<Value, ApiError> {
    state.cache.get_or(format!("miners:{hours}"), || miner_rows(state, hours)).await
}

async fn cached_validators(state: &Arc<State>, hours: i64) -> Result<Value, ApiError> {
    state.cache.get_or(format!("validators:{hours}"), || validator_rows(state, hours)).await
}

fn hours(query: &Query) -> Result<i64, ApiError> {
    int_param(query, "hours", SHARE_WINDOW_H, 1, MAX_WINDOW_H)
}

pub async fn overview(Shared(state): Shared<Arc<State>>, QueryParams(query): QueryParams<Query>) -> Answer {
    let hours = hours(&query)?;
    let made = state
        .cache
        .get_or(format!("overview:{hours}"), || async {
            let miners = cached_miners(&state, hours).await?;
            let validators = cached_validators(&state, hours).await?;
            let worked: Vec<&Value> = miners.as_array().into_iter().flatten().filter(|m| number(&m["tasks"]) != 0.0).collect();
            let validators: Vec<&Value> = validators.as_array().into_iter().flatten().collect();
            let mut window: Map<String, Value> =
                MINER_TOTALS.iter().map(|name| (name.to_string(), Value::from(worked.iter().map(|m| m[*name].as_i64().unwrap_or(0)).sum::<i64>()))).collect();
            window.insert("votes".into(), validators.iter().map(|v| v["votes"].as_i64().unwrap_or(0)).sum::<i64>().into());
            window.insert("disagreements".into(), validators.iter().map(|v| v["disagreed"].as_i64().unwrap_or(0)).sum::<i64>().into());
            let mut total = Map::new();
            total.insert("void".into(), 0.into());
            total.extend(state.reader.read(|conn| validations::verdicts(conn, None)).await?);
            let redis = &state.redis;
            let at = now();
            let mut queue = Map::new();
            for kind in KINDS {
                queue.insert(kind.into(), state.tasks(kind).depth(redis).await?.into());
            }
            let claimed: i64 = redis.clone().zcard(CLAIMS).await?;
            Ok(json!({
                "as_of": visible(&state),
                "window_hours": hours,
                "queue": queue,
                "claimed": claimed,
                "validating": state.validation.depth(redis).await? + state.validation.seeding(redis).await?,
                "oldest_validation_s": state.validation.oldest_age(redis, at).await?,
                "publishing": state.publish.depth(redis).await?,
                "miners": worked.len(),
                "validators": {"active": validators.iter().filter(|v| v["active"] == true).count(), "known": validators.len()},
                "window": window,
                "total": total,
            }))
        })
        .await?;
    Ok(Json(made))
}

pub async fn series(Shared(state): Shared<Arc<State>>, QueryParams(query): QueryParams<Query>) -> Answer {
    let bucket_minutes = int_param(&query, "bucket_minutes", 60, i64::MIN, i64::MAX)?;
    let buckets = int_param(&query, "buckets", 24, 1, MAX_BUCKETS)?;
    let (miner, validator) = (id_param(&query, "miner")?, id_param(&query, "validator")?);
    if !BUCKET_MINUTES.contains(&bucket_minutes) {
        return Err(ApiError::status(422, "bucket_minutes is one of (5, 15, 60)"));
    }
    if miner.is_some() && validator.is_some() {
        return Err(ApiError::status(422, "pass a miner or a validator, not both"));
    }
    let bucket_s = bucket_minutes * 60;
    let key = format!("series:{bucket_s}:{buckets}:{miner:?}:{validator:?}");
    let made = state
        .cache
        .get_or(key, || async {
            let until = visible(&state);
            let points = state
                .reader
                .read(move |conn| {
                    let last = (until / bucket_s as f64).floor() as i64;
                    let first = last - buckets + 1;
                    let since = (first * bucket_s) as f64;
                    let (names, found) = match &validator {
                        Some(validator) => {
                            (["votes", "pass", "fail", "void", "agreed", "disagreed"], validations::vote_series(conn, bucket_s, since, until, validator)?)
                        }
                        None => (
                            ["tasks", "pass", "fail", "void", "returned", "credited"],
                            validations::task_series(conn, bucket_s, since, until, miner.as_deref())?,
                        ),
                    };
                    Ok((first..=last)
                        .map(|bucket| {
                            let mut point = Map::new();
                            point.insert("at".into(), (bucket * bucket_s).into());
                            match found.get(&bucket) {
                                Some(Value::Object(counts)) => point.extend(counts.clone()),
                                _ => point.extend(names.iter().map(|n| (n.to_string(), Value::from(0)))),
                            }
                            Value::Object(point)
                        })
                        .collect::<Vec<_>>())
                })
                .await?;
            Ok(json!({"bucket_s": bucket_s, "points": points}))
        })
        .await?;
    Ok(Json(made))
}

async fn in_progress(state: &Arc<State>) -> Result<Value, ApiError> {
    let redis = &state.redis;
    let claimed: Vec<(String, f64)> = redis.clone().zrange_withscores(CLAIMS, 0, LIVE_ROWS - 1).await?;
    let mut claims = Vec::new();
    for (task_id, expires_at) in claimed {
        // Completed or taken back since the claims were read.
        let Some(holder) = state.claim_holder(&task_id).await? else { continue };
        let payload = state.payload(&task_id).await?.unwrap_or_else(|| json!({}));
        let urls = payload.get("url_count").cloned().unwrap_or_else(|| payload["urls"].as_array().map_or(0, Vec::len).into());
        let claim = json!({"task_id": task_id, "kind": kind_of(&payload), "miner": holder, "urls": urls, "expires_at": expires_at});
        claims.push(with_uids(state, claim.as_object().cloned().unwrap_or_default()));
    }
    let active: BTreeSet<String> = state.validation.active(redis, now()).await?.into_iter().collect();
    let mut waiting = state.validation.open_ids(redis, LIVE_ROWS).await?;
    if waiting.len() < LIVE_ROWS as usize {
        waiting.extend(state.validation.seeding_ids(redis, LIVE_ROWS - waiting.len() as isize).await?);
    }
    let mut uploads = Vec::new();
    for task_id in waiting {
        let Some(job) = state.validation.job(redis, &task_id).await? else { continue };
        let voters = state.validation.voters(redis, &task_id).await?;
        let electorate: BTreeSet<&String> = active.iter().chain(voters.iter()).collect();
        let upload = json!({
            "task_id": task_id,
            "kind": kind_of(&job),
            "miner": job["miner"],
            "urls": job["urls"].as_array().map_or(0, Vec::len),
            "completed_at": job["completed_at"],
            "deadline": job.get("deadline").cloned().unwrap_or(Value::Null),
            "voters": voters.iter().map(|v| json!({"hotkey": v, "uid": uid_of(state, Some(v))})).collect::<Vec<_>>(),
            "electorate": electorate.len(),
        });
        uploads.push(with_uids(state, upload.as_object().cloned().unwrap_or_default()));
    }
    let claims_total: i64 = redis.clone().zcard(CLAIMS).await?;
    Ok(json!({
        "claims": claims,
        "uploads": uploads,
        "claims_total": claims_total,
        "uploads_total": state.validation.depth(redis).await? + state.validation.seeding(redis).await?,
    }))
}

pub async fn live(Shared(state): Shared<Arc<State>>) -> Answer {
    Ok(Json(state.cache.get_or("live".into(), || in_progress(&state)).await?))
}

pub async fn miners(Shared(state): Shared<Arc<State>>, QueryParams(query): QueryParams<Query>) -> Answer {
    let hours = hours(&query)?;
    Ok(Json(json!({"window_hours": hours, "miners": cached_miners(&state, hours).await?})))
}

pub async fn miner(Shared(state): Shared<Arc<State>>, Path(hotkey): Path<String>) -> Answer {
    path_id(&hotkey)?;
    let until = visible(&state);
    let mut pools = Map::new();
    for kind in KINDS {
        let (asked, pool) = (hotkey.clone(), kind.to_string());
        let (budget, locked) =
            state.reader.read(move |conn| Ok((budgets::get(conn, &asked, &pool)?, budgets::locked_until(conn, &asked, &pool, now())?))).await?;
        let tasks = state.tasks(kind);
        pools.insert(
            kind.into(),
            json!({
                "budget": budget.budget,
                "verified": budget.verified,
                "in_flight": tasks.in_flight(&state.redis, &hotkey).await?,
                "waiting": tasks.waiting(&state.redis, &hotkey).await?,
                "locked_until": locked,
            }),
        );
    }
    let asked = hotkey.clone();
    let (totals, shares, verdicts, pools_of, coverage, transitions) = state
        .reader
        .read(move |conn| {
            Ok((
                validations::miner_totals(conn, until - SHARE_WINDOW_H as f64 * HOUR, until)?,
                budgets::shares(conn, SHARE_WINDOW_H, until)?,
                validations::verdicts(conn, Some(&asked))?,
                budgets::pools_of(conn, &asked)?,
                budgets::coverage_report(conn, SHARE_WINDOW_H, until)?,
                budgets::history(conn, &asked, 100)?,
            ))
        })
        .await?;
    let mut window: Map<String, Value> = MINER_TOTALS.iter().map(|name| (name.to_string(), Value::from(0))).collect();
    if let Some(Value::Object(total)) = totals.get(&hotkey) {
        window.extend(total.clone());
    }
    window.remove("last_scored_at");
    let known = verdicts.values().map(|v| v.as_i64().unwrap_or(0)).sum::<i64>() != 0 || !pools_of.is_empty();
    let mut page = json!({
        "known": known,
        "window_hours": SHARE_WINDOW_H,
        "pools": pools,
        "coverage": coverage.get(&hotkey).cloned().unwrap_or_else(|| json!({})),
        "verdicts": verdicts,
        "share": shares.get(CRAWL).and_then(|s| s.get(&hotkey)).copied().unwrap_or(0.0),
        "window": window,
        "transitions": transitions,
    });
    let mut keys = with_keys(&state, Map::from_iter([("hotkey".to_string(), Value::from(hotkey.as_str()))]), &hotkey).as_object().cloned().unwrap_or_default();
    keys.extend(page.as_object_mut().map(std::mem::take).unwrap_or_default());
    Ok(Json(Value::Object(keys)))
}

pub async fn validators(Shared(state): Shared<Arc<State>>, QueryParams(query): QueryParams<Query>) -> Answer {
    let hours = hours(&query)?;
    Ok(Json(json!({"window_hours": hours, "validators": cached_validators(&state, hours).await?})))
}

pub async fn validator(Shared(state): Shared<Arc<State>>, Path(hotkey): Path<String>, QueryParams(query): QueryParams<Query>) -> Answer {
    path_id(&hotkey)?;
    let hours = hours(&query)?;
    let rows = cached_validators(&state, hours).await?;
    let found = rows.as_array().into_iter().flatten().find(|row| row["hotkey"] == hotkey.as_str()).and_then(Value::as_object).cloned();
    let (row, known) = match found {
        Some(row) => (row, true),
        None => (validator_row(&hotkey, None, None, &HashMap::new()), false),
    };
    let mut page = with_keys(&state, row, &hotkey).as_object().cloned().unwrap_or_default();
    page.insert("known".into(), known.into());
    Ok(Json(Value::Object(page)))
}

pub async fn votes(Shared(state): Shared<Arc<State>>, QueryParams(query): QueryParams<Query>) -> Answer {
    let filter = VoteFilter {
        validator: id_param(&query, "validator")?,
        miner: id_param(&query, "miner")?,
        task_id: id_param(&query, "task_id")?,
        verdict: literal_param(&query, "verdict", &VERDICTS)?,
        agreed: bool_param(&query, "agreed")?,
        until: Some(visible(&state)),
        before: optional_int(&query, "before")?,
        limit: int_param(&query, "limit", PAGE, 1, MAX_PAGE)?,
    };
    let (found, next) = state.reader.read(move |conn| validations::votes(conn, &filter)).await?;
    let votes: Vec<Value> = found.into_iter().map(|vote| with_uids(&state, vote.as_object().cloned().unwrap_or_default())).collect();
    Ok(Json(json!({"votes": votes, "next": next})))
}

pub async fn events(Shared(state): Shared<Arc<State>>, QueryParams(query): QueryParams<Query>) -> Answer {
    let miner = id_param(&query, "miner")?;
    let task_id = id_param(&query, "task_id")?;
    let outcome = literal_param(&query, "outcome", &["issued", "completed", "refused", "reclaimed", "dropped"])?;
    let before = optional_int(&query, "before")?;
    let limit = int_param(&query, "limit", PAGE, 1, MAX_PAGE)?;
    let (found, next) = state.reader.read(move |conn| roundlog::events(conn, miner.as_deref(), task_id.as_deref(), outcome.as_deref(), before, limit)).await?;
    Ok(Json(json!({"events": found, "next": next})))
}

/// Finalized tasks, newest first; pass `next` back as `before` for the next page.
pub async fn tasks(Shared(state): Shared<Arc<State>>, QueryParams(query): QueryParams<Query>) -> Answer {
    let shown = visible(&state);
    let filter = TaskFilter {
        miner: id_param(&query, "miner")?,
        validator: id_param(&query, "validator")?,
        verdict: literal_param(&query, "verdict", &VERDICTS)?,
        kind: literal_param(&query, "kind", &KINDS)?,
        since: float_param(&query, "since")?.unwrap_or(0.0),
        before: Some(float_param(&query, "before")?.map_or(shown, |before| before.min(shown))),
        limit: int_param(&query, "limit", PAGE, 1, MAX_PAGE)?,
    };
    let limit = filter.limit;
    let found = state.reader.read(move |conn| validations::recent(conn, &filter)).await?;
    let next = (found.len() as i64 == limit).then(|| found.last().map(|t| t["scored_at"].clone())).flatten();
    let tasks: Vec<Value> = found.into_iter().map(|task| with_uids(&state, task)).collect();
    Ok(Json(json!({"tasks": tasks, "next": next})))
}

pub async fn task(Shared(state): Shared<Arc<State>>, Path(task_id): Path<String>) -> Answer {
    path_id(&task_id)?;
    let shown = visible(&state);
    let asked = task_id.clone();
    let uploads: Vec<Map<String, Value>> = state
        .reader
        .read(move |conn| validations::uploads(conn, &asked))
        .await?
        .into_iter()
        .filter(|u| number(&u["scored_at"]) <= shown)
        .map(|u| with_uids(&state, u).as_object().cloned().unwrap_or_default())
        .collect();
    let scored = uploads.first().cloned();
    let mut page = with_uids(&state, task_state(&state, &task_id, scored.as_ref()).await?).as_object().cloned().unwrap_or_default();
    let Some(scored) = scored else {
        page.extend([("uploads".to_string(), json!([])), ("votes".to_string(), json!([])), ("urls".to_string(), json!([]))]);
        return Ok(Json(Value::Object(page)));
    };
    let (asked, key) = (task_id.clone(), text(&Value::Object(scored.clone()), "upload_key").to_string());
    let (votes, urls) = state.reader.read(move |conn| Ok((validations::votes_on(conn, &asked, &key)?, validations::urls(conn, &asked)?))).await?;
    page.insert("uploads".into(), uploads.into_iter().map(Value::Object).collect::<Vec<_>>().into());
    page.insert("votes".into(), votes.into_iter().map(|v| with_uids(&state, v.as_object().cloned().unwrap_or_default())).collect::<Vec<_>>().into());
    page.insert("urls".into(), urls);
    Ok(Json(Value::Object(page)))
}

async fn task_state(state: &State, task_id: &str, scored: Option<&Map<String, Value>>) -> Result<Map<String, Value>, ApiError> {
    let score = scored.cloned().map_or(Value::Null, Value::Object);
    let state_of = |status: &str, round_id: Value, miner: Value| -> Map<String, Value> {
        json!({"task_id": task_id, "status": status, "round_id": round_id, "miner": miner, "score": score}).as_object().cloned().unwrap_or_default()
    };
    if let Some(job) = state.validation.job(&state.redis, task_id).await? {
        let status = if state.validation.voters(&state.redis, task_id).await?.is_empty() { "open" } else { "voting" };
        return Ok(state_of(status, job["round_id"].clone(), job["miner"].clone()));
    }
    if let Some(payload) = state.payload(task_id).await? {
        let holder = state.claim_holder(task_id).await?;
        let status = if holder.is_some() { "claimed" } else { "queued" };
        return Ok(state_of(status, payload.get("round_id").cloned().unwrap_or(Value::Null), holder.map_or(Value::Null, Value::from)));
    }
    let Some(scored) = scored else {
        return Err(ApiError::status(404, "no such task"));
    };
    Ok(state_of(text(&Value::Object(scored.clone()), "verdict"), scored["round_id"].clone(), scored["miner"].clone()))
}
