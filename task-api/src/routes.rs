//! Claims, completions and verdicts, the bot's enqueue and room, and the public round pages.

use std::net::SocketAddr;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Bytes;
use axum::extract::{ConnectInfo, Path, Query as QueryParams, State as Shared};
use axum::http::{HeaderMap, Uri};
use axum::response::{IntoResponse, Response};
use axum::Json;
use desearch::r2::{Error as StorageError, PARQUET};
use desearch::time::{now, utc_day};
use rand::Rng;
use redis::AsyncCommands;
use serde_json::{json, Map, Value};

use crate::auth::{self, client_address, Caller};
use crate::budgets::{self, CRAWL, EMBED, SHARE_WINDOW_H, WAITING_PER_BUDGET};
use crate::http::{json_body, unavailable, Answer, ApiError, Query, CLAIM_PATH};
use crate::lifecycle::{self, kind_of, signed_manifest, text, EMBED_FIELDS};
use crate::manifest::REVEAL_AFTER_BLOCKS;
use crate::models::{self, ClaimBody, CompleteBody, Enqueue};
use crate::py::round_to;
use crate::queues::{self, Claim, Refusal, CLAIMS};
use crate::roundlog::{self, Receipt};
use crate::rounds::UPLOAD_GRACE_S;
use crate::state::State;
use crate::storage::{delete_quietly, is_parquet, put_json};
use crate::validations::{self, TaskFilter};
use crate::verdicts::{build_embed_vote, build_vote};
use crate::{roundstore, sampling, uploadlog};

const COMPLETE_LOCK_S: i64 = 300;
const COMPLETED_TTL_S: u64 = 900;
const COMPLETE_WAIT: Duration = Duration::from_secs(20);
const COMPLETE_POLL: Duration = Duration::from_millis(250);
/// R2's clock and ours may differ by this much.
const CLOCK_SLACK_S: f64 = 2.0;
const SLOW_COMPLETE_S: f64 = 5.0;
const LINK_SLACK_S: u64 = 60;
const TASKS_PAGE: i64 = 50;
const MAX_TASKS_PAGE: i64 = 100;
const ENQUEUED_TTL_S: u64 = 86_400;

/// What a refused miner should wait, so idle polling does not fill the signed log.
fn retry_after_for(code: &str) -> f64 {
    match code {
        "QUEUE_EMPTY" | "ALREADY_HELD" | "WAITING_FOR_VERDICTS" => 10.0,
        "NO_CAPACITY" => 5.0,
        "VALIDATION_BACKLOG" => 30.0,
        "KIND_CLOSED" => 3600.0,
        _ => 5.0,
    }
}

/// The method, path, body and address a signed request came with.
pub struct Signed<'a> {
    pub headers: &'a HeaderMap,
    pub method: &'a str,
    pub uri: &'a Uri,
    pub body: &'a [u8],
    pub peer: SocketAddr,
}

impl Signed<'_> {
    pub async fn caller(&self, state: &State) -> Result<Caller, ApiError> {
        let address = client_address(self.headers, &self.peer.ip().to_string());
        auth::caller(state, self.headers, self.method, self.uri.path(), self.body, &address).await
    }
}

async fn validator(state: &State, signed: &Signed<'_>) -> Result<Caller, ApiError> {
    let who = signed.caller(state).await?;
    if !who.is_validator {
        return Err(ApiError::status(403, "only validators may do this"));
    }
    Ok(who)
}

pub async fn claim(Shared(state): Shared<Arc<State>>, ConnectInfo(peer): ConnectInfo<SocketAddr>, headers: HeaderMap, uri: Uri, body: Bytes) -> Response {
    let signed = Signed { headers: &headers, method: "POST", uri: &uri, body: &body, peer };
    match claim_tasks(&state, &signed).await {
        Ok(answer) => answer.into_response(),
        Err(ApiError::Unavailable) => unavailable(CLAIM_PATH),
        Err(error) => error.into_response(),
    }
}

async fn claim_tasks(state: &State, signed: &Signed<'_>) -> Answer {
    let who = signed.caller(state).await?;
    if who.is_validator || who.is_admin {
        return Err(ApiError::status(403, "validators may not claim tasks"));
    }
    let body = ClaimBody::parse(&json_body(signed.body)?)?;
    let kind: &'static str = if body.kind == EMBED { EMBED } else { CRAWL };
    let round_id = state.current_round(kind);

    if let Some(retry_after) = state.retry_after(&who.hotkey).await? {
        // Not logged: signing and storing every excess poll would let a flooder grow the log.
        let refusal = json!({"code": "RATE_LIMITED", "inputs": {"per_sec": state.settings.poll_rate, "retry_after": retry_after}});
        return Ok(Json(json!({"tasks": [], "refusal": refusal, "receipt": null})));
    }
    if kind == EMBED && !state.settings.embed_tasks {
        return refused(state, &round_id, &who, Refusal::new("KIND_CLOSED", json!({"kind": kind}))).await;
    }
    let at = now();
    let (hotkey, pool) = (who.hotkey.clone(), kind.to_string());
    if let Some(until) = state.db.run(move |conn| budgets::locked_until(conn, &hotkey, &pool, at)).await? {
        let inputs = json!({"until": round_to(until, 3), "retry_after": round_to(until - now(), 3)});
        return refused(state, &round_id, &who, Refusal::new("LOCKED_OUT", inputs)).await;
    }
    // Stop leasing work whose upload would expire before validation.
    let validating = state.validation.oldest_age(&state.redis, at).await?.max(state.validation.oldest_seeding_age(&state.redis, at).await?);
    let publishing = state.publish.oldest_age(&state.redis, at).await?;
    let waiting = state.publish.waiting(&state.redis).await?;
    let max_backlog = state.settings.max_backlog;
    if state.publish_rate.lock().expect("publish rate").overloaded(waiting) || (max_backlog != 0.0 && validating.max(publishing) > max_backlog) {
        let inputs = json!({"validation_s": validating, "publish_s": publishing, "limit_s": max_backlog, "publish_waiting": waiting});
        return refused(state, &round_id, &who, Refusal::new("VALIDATION_BACKLOG", inputs)).await;
    }
    let (hotkey, pool) = (who.hotkey.clone(), kind.to_string());
    let budget = state.db.run(move |conn| budgets::get_or_create(conn, &hotkey, &pool)).await?.budget;
    let tasks = state.tasks(kind);
    let claims = match tasks.claim(&state.redis, &who.hotkey, budget, WAITING_PER_BUDGET * budget, body.count, now()).await? {
        Ok(claims) => claims,
        Err(refusal) => return refused(state, &round_id, &who, refusal).await,
    };
    let mut issued = Vec::new();
    let mut receipts = Vec::new();
    for got in claims {
        if let Some(task) = issue(state, &got, kind, &who).await? {
            receipts.push(Receipt {
                round_id: text(&task, "round_id").into(),
                hotkey: who.hotkey.clone(),
                requested_at: who.requested_at,
                outcome: "issued",
                seq: got.seq,
                task_id: Some(got.task_id.clone()),
                ..Receipt::default()
            });
            issued.push(task);
        }
    }
    if issued.is_empty() {
        return Err(ApiError::status(503, "upload signing is unavailable"));
    }
    let receipts = state.record(receipts).await?;
    // The time spent signing and recording the claims is ours, not the miner's.
    let task_ids: Vec<String> = issued.iter().map(|task| text(task, "task_id").to_string()).collect();
    let expiry = tasks.start_clocks(&state.redis, &who.hotkey, &task_ids, now()).await?;
    for task in &mut issued {
        task["expires_at"] = (expiry - UPLOAD_GRACE_S).into();
        task["upload"]["expires_at"] = (expiry - UPLOAD_GRACE_S).into();
    }
    Ok(Json(json!({"tasks": issued, "receipts": receipts})))
}

async fn refused(state: &State, round_id: &str, who: &Caller, mut refusal: Refusal) -> Answer {
    refusal.inputs.entry("retry_after").or_insert_with(|| retry_after_for(refusal.code).into());
    let refusal_value = refusal.as_value();
    if !state.first_refusal(&who.hotkey, refusal.code).await? {
        return Ok(Json(json!({"tasks": [], "refusal": refusal_value, "receipt": null})));
    }
    let seq = state.next_seq().await?;
    let receipt = state
        .record_one(Receipt {
            round_id: round_id.into(),
            hotkey: who.hotkey.clone(),
            requested_at: who.requested_at,
            outcome: "refused",
            seq,
            refusal: Some(refusal_value.clone()),
            ..Receipt::default()
        })
        .await?;
    Ok(Json(json!({"tasks": [], "refusal": refusal_value, "receipt": receipt})))
}

/// The task as the miner receives it, with an upload link only it can use.
async fn issue(state: &State, got: &Claim, kind: &str, who: &Caller) -> Result<Option<Value>, ApiError> {
    let task_id = &got.task_id;
    let name = format!("task={task_id}/{}-{}.parquet", who.hotkey, got.seq);
    let upload_key = format!("uploads/dt={}/{name}", utc_day(now()));
    let since_epoch = Duration::from_secs_f64(now());
    // It may outlive the claim; a late completion or a file written after it is refused.
    let upload_url = state.storage.presign("PUT", &upload_key, state.settings.claim_ttl as u64 + LINK_SLACK_S, Some(PARQUET), since_epoch);
    let inputs = if kind == EMBED {
        let payload = &got.payload;
        match (payload.get("model"), payload.get("texts"), payload.get("input_key").and_then(Value::as_str), payload.get("input_sha256")) {
            (Some(model), Some(texts), Some(input_key), Some(sha256)) => {
                let url = state.storage.presign("GET", input_key, state.settings.claim_ttl as u64, None, since_epoch);
                json!({"model": model, "texts": texts, "input": {"url": url, "sha256": sha256}})
            }
            _ => {
                eprintln!("could not presign the upload for {task_id}: its embed inputs are missing");
                state.tasks(kind).abandon(&state.redis, task_id, &who.hotkey).await?;
                return Ok(None);
            }
        }
    } else {
        json!({})
    };
    let issued = json!({"hotkey": who.hotkey, "key": upload_key, "name": name, "at": now()});
    let _: () = state.redis.clone().set(format!("issued:{task_id}"), issued.to_string()).await?;
    let expires_at = got.expires_at - UPLOAD_GRACE_S;
    let round_id = got.payload.get("round_id").cloned().unwrap_or_else(|| state.current_round(kind).into());
    let mut task = Map::new();
    task.insert("task_id".into(), task_id.clone().into());
    task.insert("kind".into(), kind.into());
    task.insert("round_id".into(), round_id);
    task.insert("expires_at".into(), expires_at.into());
    task.insert("urls".into(), got.payload.get("urls").cloned().unwrap_or_else(|| json!([])));
    task.extend(inputs.as_object().cloned().unwrap_or_default());
    task.insert("upload".into(), json!({"url": upload_url, "key": upload_key, "content_type": PARQUET, "expires_at": expires_at}));
    Ok(Some(Value::Object(task)))
}

async fn completed_by(state: &State, task_id: &str, hotkey: &str) -> Result<Option<Value>, ApiError> {
    let done: Option<String> = state.redis.clone().get(format!("completed:{task_id}")).await?;
    let done: Option<Value> = done.and_then(|raw| serde_json::from_str(&raw).ok());
    Ok(done.filter(|d| d["miner"] == hotkey).map(|d| d["result"].clone()))
}

/// A repeated call waits for the running one instead of failing, so a retry is never an abandon.
async fn hold_completion(state: &State, task_id: &str, hotkey: &str) -> Result<String, ApiError> {
    let lock = format!("completing:{task_id}");
    let waited = Instant::now();
    while !state.set_once(&lock, hotkey, COMPLETE_LOCK_S).await? {
        if waited.elapsed() > COMPLETE_WAIT {
            return Err(ApiError::retry(503, "a completion for this task is already running", "2"));
        }
        tokio::time::sleep(COMPLETE_POLL).await;
    }
    Ok(lock)
}

pub async fn complete(
    Shared(state): Shared<Arc<State>>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    Path(task_id): Path<String>,
    headers: HeaderMap,
    uri: Uri,
    body: Bytes,
) -> Answer {
    // The deadline is judged when the request arrived, not after our own work on it.
    let arrived = now();
    let signed = Signed { headers: &headers, method: "POST", uri: &uri, body: &body, peer };
    let who = signed.caller(&state).await?;
    let report = CompleteBody::parse(&json_body(&body)?)?;
    let holds = state.claim_holder(&task_id).await?.as_deref() == Some(who.hotkey.as_str());
    if !holds && completed_by(&state, &task_id, &who.hotkey).await?.is_none() {
        return Err(ApiError::status(409, "you do not hold this claim"));
    }
    let lock = hold_completion(&state, &task_id, &who.hotkey).await?;
    let outcome = finish_completion(&state, &task_id, &report, &who, arrived).await;
    state.delete(&lock).await?;
    outcome.map(Json)
}

async fn finish_completion(state: &State, task_id: &str, report: &CompleteBody, who: &Caller, arrived: f64) -> Result<Value, ApiError> {
    if let Some(done) = completed_by(state, task_id, &who.hotkey).await? {
        return Ok(done);
    }
    if state.claim_holder(task_id).await?.as_deref() != Some(who.hotkey.as_str()) {
        return Err(ApiError::status(409, "you do not hold this claim"));
    }
    match queues::claim_expiry(&state.redis, task_id).await? {
        Some(expiry) if expiry >= arrived => {}
        _ => return Err(ApiError::status(409, "you do not hold a live claim on this task")),
    }
    let result = verify_completion(state, task_id, report, who, arrived).await?;
    let done = json!({"miner": who.hotkey, "result": result});
    let _: () = state.redis.clone().set_ex(format!("completed:{task_id}"), done.to_string(), COMPLETED_TTL_S).await?;
    Ok(result)
}

fn storage_down(what: &str, error: StorageError) -> ApiError {
    eprintln!("{what}: {error}");
    ApiError::status(502, "object storage is unavailable, retry")
}

async fn verify_completion(state: &State, task_id: &str, report: &CompleteBody, who: &Caller, arrived: f64) -> Result<Value, ApiError> {
    let issued: Option<String> = state.redis.clone().get(format!("issued:{task_id}")).await?;
    let issued: Value = issued.and_then(|raw| serde_json::from_str(&raw).ok()).unwrap_or_else(|| json!({}));
    if issued.get("key").and_then(Value::as_str) != Some(report.key.as_str()) {
        return Err(ApiError::status(400, "key is not the one issued with this claim"));
    }
    let found = state.storage.head(&report.key).await.map_err(|e| storage_down(&format!("HEAD failed for {}", report.key), e))?;
    let Some(found) = found.filter(|f| f.size > 0) else {
        return Err(ApiError::status(422, "nothing was uploaded to the issued key"));
    };
    if found.modified.is_some_and(|modified| modified as f64 > arrived + CLOCK_SLACK_S) {
        return Err(ApiError::status(409, "the upload was written after this completion"));
    }
    if found.size > state.settings.max_upload {
        delete_quietly(&state.storage, &report.key).await;
        return Err(ApiError::status(413, format!("upload is {} bytes, the limit is {}", found.size, state.settings.max_upload)));
    }
    let framed =
        is_parquet(&state.storage, &report.key, found.size).await.map_err(|e| storage_down(&format!("could not read the ends of {}", report.key), e))?;
    if !framed {
        return Err(ApiError::status(422, "the upload is not a Parquet file"));
    }

    // The PUT URL outlives this call, so work on a copy the miner can't touch.
    let attempt: String = (0..4).map(|_| format!("{:02x}", rand::thread_rng().gen::<u8>())).collect();
    let name = text(&issued, "name");
    let frozen = format!("submitted/dt={}/{}-{attempt}.parquet", utc_day(now()), name.strip_suffix(".parquet").unwrap_or(name));
    let stat_s = now() - arrived;
    let frozen_etag = match state.storage.copy(&report.key, &frozen, Some(&found.etag)).await {
        Ok(etag) => etag,
        Err(StorageError::Changed) => return Err(ApiError::status(409, "the upload changed while completing, retry")),
        Err(error) => return Err(storage_down(&format!("could not freeze {}", report.key), error)),
    };

    let payload = state.payload(task_id).await?.unwrap_or_else(|| json!({}));
    let kind = kind_of(&payload).to_string();
    let round_id = Some(text(&payload, "round_id")).filter(|r| !r.is_empty()).map(String::from).unwrap_or_else(|| state.current_round(&kind));
    // The sample seed is a block hash nobody knew when the upload was frozen.
    let frozen_block = state.seeds.current_block().await?;
    let completed_at = arrived;
    let position = payload.get("position").cloned().unwrap_or_else(|| 0.into());
    let mut job = json!({
        "task_id": task_id,
        "kind": kind,
        "round_id": round_id,
        "miner": who.hotkey,
        "key": frozen,
        "etag": frozen_etag,
        "urls": payload.get("urls").cloned().unwrap_or_else(|| json!([])),
        "position": position,
        "rank": payload.get("rank").cloned().unwrap_or(position),
        "attempts": payload.get("attempts").cloned().unwrap_or_else(|| 0.into()),
        "size": found.size,
        "reported": report.reported,
        "claimed_at": issued.get("at").cloned().unwrap_or(Value::Null),
        "completed_at": completed_at,
        "frozen_block": frozen_block,
        "seed_block": frozen_block + REVEAL_AFTER_BLOCKS,
        "deadline": completed_at + state.seeds.wait_s() + state.settings.validation_ttl,
    });
    for name in EMBED_FIELDS {
        if let Some(value) = payload.get(name) {
            job[name] = value.clone();
        }
    }
    job["manifest"] = signed_manifest(state, &job);
    if let Err(error) = put_json(&state.storage, &lifecycle::manifest_key(&frozen), &job["manifest"], None).await {
        delete_quietly(&state.storage, &frozen).await;
        return Err(storage_down(&format!("could not write the manifest for {frozen}"), error));
    }
    let Some(seq) = state.tasks(&kind).complete(&state.redis, task_id, &who.hotkey, &job, &report.key, arrived).await? else {
        delete_quietly(&state.storage, &frozen).await;
        return Err(ApiError::status(409, "you do not hold a live claim on this task"));
    };
    delete_quietly(&state.storage, &report.key).await;
    let took = now() - arrived;
    if took > SLOW_COMPLETE_S {
        eprintln!("completing {task_id} took {took:.1}s ({stat_s:.1}s before the copy)");
    }
    if kind == CRAWL {
        sampling::note_upload(&state.redis, &who.hotkey, completed_at).await?;
        uploadlog::note(&state.redis, &job).await?;
    }
    state
        .record_one(Receipt {
            round_id,
            hotkey: who.hotkey.clone(),
            requested_at: who.requested_at,
            outcome: "completed",
            seq,
            task_id: Some(task_id.into()),
            block: Some(frozen_block),
            ..Receipt::default()
        })
        .await?;
    Ok(json!({"task_id": task_id, "status": "open_for_validation"}))
}

pub async fn abandon(
    Shared(state): Shared<Arc<State>>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    Path(task_id): Path<String>,
    headers: HeaderMap,
    uri: Uri,
    body: Bytes,
) -> Answer {
    let who = Signed { headers: &headers, method: "POST", uri: &uri, body: &body, peer }.caller(&state).await?;
    let payload = state.payload(&task_id).await?.unwrap_or_else(|| json!({}));
    let kind = kind_of(&payload).to_string();
    if state.claim_holder(&task_id).await?.as_deref() != Some(who.hotkey.as_str()) {
        return Err(ApiError::status(409, "you do not hold this claim"));
    }
    // A completion still running decides first; a finished one leaves nothing to abandon.
    let lock = hold_completion(&state, &task_id, &who.hotkey).await?;
    let seq = state.tasks(&kind).abandon(&state.redis, &task_id, &who.hotkey).await;
    state.delete(&lock).await?;
    let Some(seq) = seq? else {
        return Err(ApiError::status(409, "you do not hold this claim"));
    };
    let round_id = Some(text(&payload, "round_id")).filter(|r| !r.is_empty()).map(String::from).unwrap_or_else(|| state.current_round(&kind));
    state
        .record_one(Receipt {
            round_id,
            hotkey: who.hotkey.clone(),
            requested_at: who.requested_at,
            outcome: "reclaimed",
            seq,
            task_id: Some(task_id.clone()),
            cause: Some("abandoned".into()),
            ..Receipt::default()
        })
        .await?;
    let (hotkey, task, lapsed) = (who.hotkey.clone(), task_id.clone(), payload);
    let budget = state.db.run(move |conn| lifecycle::lapse(conn, &hotkey, &task, &lapsed, "abandoned", now())).await?;
    Ok(Json(json!({"task_id": task_id, "budget": budget})))
}

/// An upload storage lost is void; nothing else is anyone's to hand back.
pub async fn release(
    Shared(state): Shared<Arc<State>>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    Path(task_id): Path<String>,
    headers: HeaderMap,
    uri: Uri,
    body: Bytes,
) -> Answer {
    let who = validator(&state, &Signed { headers: &headers, method: "POST", uri: &uri, body: &body, peer }).await?;
    models::release(&json_body(&body)?)?;
    let job = state.validation.job(&state.redis, &task_id).await?;
    let Some(job) = job.filter(|job| job.get("picked").is_some_and(|p| !p.is_null() && p != false && p != "")) else {
        return Err(ApiError::status(409, "no such open upload"));
    };
    let gone = state.storage.head(text(&job, "key")).await.map_err(|_| ApiError::status(502, "object storage is unavailable, retry"))?.is_none();
    if !gone {
        return Ok(Json(json!({"task_id": task_id, "status": "open"})));
    }
    let Some(lapsed) = state.validation.finalize(&state.redis, &task_id, None, false, now()).await? else {
        return Err(ApiError::status(409, "the upload was finalized"));
    };
    lifecycle::void_task(&state, &task_id, lapsed, &who.hotkey, "upload_missing").await?;
    Ok(Json(json!({"task_id": task_id, "status": "void"})))
}

pub async fn score(
    Shared(state): Shared<Arc<State>>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    Path(task_id): Path<String>,
    headers: HeaderMap,
    uri: Uri,
    body: Bytes,
) -> Answer {
    let who = validator(&state, &Signed { headers: &headers, method: "POST", uri: &uri, body: &body, peer }).await?;
    let result = match json_body(&body)? {
        Value::Object(result) => result,
        Value::Null => return Err(models::Invalid::single("missing", &["body"], "Field required").into()),
        _ => return Err(models::Invalid::single("dict_type", &["body"], "Input should be a valid dictionary").into()),
    };
    let lock = lifecycle::lock_key(&task_id);
    if !state.set_once(&lock, &who.hotkey, lifecycle::FINALIZE_LOCK_S).await? {
        return Err(ApiError::retry(503, "a verdict for this upload is already being recorded", "2"));
    }
    let outcome = record_verdict(&state, &task_id, result, &who).await;
    state.delete(&lock).await?;
    outcome.map(Json)
}

async fn record_verdict(state: &State, task_id: &str, result: Map<String, Value>, who: &Caller) -> Result<Value, ApiError> {
    let hotkey = who.hotkey.clone();
    if state.db.run(move |conn| validations::is_excluded(conn, &hotkey)).await? {
        state.validation.leave(&state.redis, &who.hotkey).await?;
        return Err(ApiError::status(403, "this validator disagreed with too many audits"));
    }
    state.validation.present(&state.redis, &who.hotkey, now()).await?;
    let Some(job) = state.validation.job(&state.redis, task_id).await? else {
        return Err(ApiError::status(409, "no such open upload"));
    };
    let kind = kind_of(&job).to_string();
    let built =
        if kind == EMBED { build_embed_vote(&job, &who.hotkey, models::embed_score(&result)?) } else { build_vote(&job, &who.hotkey, models::score(&result)?) };
    let mut vote = built.map_err(|why| ApiError::status(422, why.0))?;
    vote["at"] = now().into();
    if state.validation.vote(&state.redis, task_id, &who.hotkey, &vote, now()).await? == 0 {
        return Err(ApiError::status(409, "this validator already voted on this upload"));
    }
    if let Some(finalized) = lifecycle::finalize_task(state, task_id, &job, now()).await? {
        return Ok(finalized);
    }
    let (miner, pool) = (text(&job, "miner").to_string(), kind.clone());
    let budget = state.db.run(move |conn| budgets::get(conn, &miner, &pool)).await?.budget;
    Ok(json!({"task_id": task_id, "verdict": "pending", "credited": 0, "miner_budget": budget}))
}

pub async fn shares(Shared(state): Shared<Arc<State>>) -> Answer {
    let as_of = now() - state.settings.ledger_delay;
    let pools = state.db.run(move |conn| budgets::shares(conn, SHARE_WINDOW_H, as_of)).await?;
    Ok(Json(json!({"window_hours": SHARE_WINDOW_H, "as_of": as_of, "pools": pools})))
}

/// How many tasks the queue can take now, for the bot that fills it.
pub async fn room(Shared(state): Shared<Arc<State>>) -> Answer {
    let at = now();
    let depth = state.crawl.depth(&state.redis).await?;
    let unrevealed = state.db.run(roundstore::unrevealed_tasks).await?;
    let claimed: i64 = state.redis.clone().zcard(CLAIMS).await?;
    // Everything not yet published, so the queue fills only as fast as the publisher empties it.
    let in_system = depth
        + unrevealed
        + claimed
        + state.validation.seeding(&state.redis).await?
        + state.validation.depth(&state.redis).await?
        + state.publish.waiting(&state.redis).await?;
    let max_backlog = state.settings.max_backlog;
    let oldest = state
        .validation
        .oldest_age(&state.redis, at)
        .await?
        .max(state.validation.oldest_seeding_age(&state.redis, at).await?)
        .max(state.publish.oldest_age(&state.redis, at).await?);
    let refusing = max_backlog != 0.0 && oldest > max_backlog;
    let (room, per_minute) = {
        let rate = state.publish_rate.lock().expect("publish rate");
        ((state.settings.queue_target - depth - unrevealed).max(0).min(rate.room(in_system)), rate.per_second() * 60.0)
    };
    Ok(Json(json!({
        "room_tasks": if refusing { 0 } else { room },
        "queue": depth,
        "unrevealed": unrevealed,
        "in_system": in_system,
        "published_per_min": round_to(per_minute, 1),
        "refusing": refusing,
    })))
}

/// A batch sent twice, after a timeout, is queued once.
pub async fn enqueue(Shared(state): Shared<Arc<State>>, ConnectInfo(peer): ConnectInfo<SocketAddr>, headers: HeaderMap, uri: Uri, body: Bytes) -> Answer {
    let who = Signed { headers: &headers, method: "POST", uri: &uri, body: &body, peer }.caller(&state).await?;
    if !who.is_admin {
        return Err(ApiError::status(403, "only admins may do this"));
    }
    let request = Enqueue::parse(&json_body(&body)?)?;
    let seen = (!request.batch_id.is_empty()).then(|| format!("enqueued:{}", request.batch_id));
    if let Some(seen) = &seen {
        let earlier: Option<String> = state.redis.clone().get(seen).await?;
        if let Some(earlier) = earlier.and_then(|raw| serde_json::from_str::<Value>(&raw).ok()) {
            return Ok(Json(earlier));
        }
    }
    let round = lifecycle::open_round(&state, request.urls).await?.map_err(|invalid| ApiError::status(422, invalid.to_string()))?;
    let found = json!({"round_id": round.round_id, "batches": round.batches.len(), "seed_block": round.seed_block, "manifest_hash": round.manifest_hash});
    if let Some(seen) = &seen {
        let _: () = state.redis.clone().set_ex(seen, found.to_string(), ENQUEUED_TTL_S).await?;
    }
    Ok(Json(found))
}

pub async fn key(Shared(state): Shared<Arc<State>>) -> Answer {
    Ok(Json(json!({"signer": state.signer()})))
}

pub async fn round_list(Shared(state): Shared<Arc<State>>, QueryParams(query): QueryParams<Query>) -> Answer {
    let limit = crate::logs::int_param(&query, "limit", 100, 1, 1000)?;
    let rounds = state.db.run(move |conn| roundstore::recent(conn, limit)).await?;
    Ok(Json(json!({"rounds": rounds})))
}

pub async fn round_view(Shared(state): Shared<Arc<State>>, Path(round_id): Path<String>) -> Answer {
    let Some(found) = state.db.run(move |conn| roundstore::get(conn, &round_id)).await? else {
        return Err(ApiError::status(404, "no such round"));
    };
    let mut view = found.public_view();
    view.insert("signer".into(), state.signer().into());
    Ok(Json(Value::Object(view)))
}

pub async fn round_log(Shared(state): Shared<Arc<State>>, Path(round_id): Path<String>) -> Answer {
    let asked = round_id.clone();
    let (entries, anchor) = state.db.run(move |conn| Ok((roundlog::entries(conn, &asked)?, roundlog::anchored_root(conn, &asked)?))).await?;
    Ok(Json(json!({"round_id": round_id, "entries": entries, "anchor_root": anchor})))
}

/// A miner's own finalized verdicts, without the public ledger's delay.
pub async fn own_verdicts(
    Shared(state): Shared<Arc<State>>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    Path(hotkey): Path<String>,
    QueryParams(query): QueryParams<Query>,
    headers: HeaderMap,
    uri: Uri,
) -> Answer {
    let since = crate::logs::float_param(&query, "since")?.unwrap_or(0.0);
    let limit = crate::logs::int_param(&query, "limit", TASKS_PAGE, 1, MAX_TASKS_PAGE)?;
    let who = Signed { headers: &headers, method: "GET", uri: &uri, body: &[], peer }.caller(&state).await?;
    if who.hotkey != hotkey {
        return Err(ApiError::status(403, "only the miner itself may read this"));
    }
    let filter = TaskFilter { miner: Some(hotkey), since, limit, ..TaskFilter::default() };
    let tasks = state.db.run(move |conn| validations::recent(conn, &filter)).await?;
    Ok(Json(json!({"tasks": tasks})))
}

pub async fn ping(Shared(state): Shared<Arc<State>>) -> Answer {
    if state.draining.load(Ordering::Relaxed) {
        return Err(ApiError::status(503, "shutting down"));
    }
    Ok(Json(json!({"ok": true})))
}

pub async fn health(Shared(state): Shared<Arc<State>>) -> Answer {
    let at = now();
    let redis = &state.redis;
    let outcomes: Option<i64> = redis.clone().get(desearch::feeds::OUTCOMES.counter()).await?;
    let mut active = state.validation.active(redis, at).await?;
    active.sort();
    let accounts = state
        .db
        .run(move |conn| {
            Ok(json!({
                "verdicts": validations::verdicts(conn, None)?,
                "validators": validations::audit_standing(conn)?,
                "pools": budgets::shares(conn, SHARE_WINDOW_H, at)?,
                "coverage": budgets::coverage_report(conn, SHARE_WINDOW_H, at)?,
            }))
        })
        .await?;
    let current = state.current.lock().expect("current rounds").clone();
    Ok(Json(json!({
        "queue_depth": {"crawl": state.crawl.depth(redis).await?, "embed": state.embed.depth(redis).await?},
        "validation_depth": state.validation.depth(redis).await?,
        "seeding": state.validation.seeding(redis).await?,
        "outcomes": outcomes.unwrap_or(0),
        "active_validators": active,
        "oldest_validation_s": state.validation.oldest_age(redis, at).await?,
        "publishing": state.publish.depth(redis).await?,
        "oldest_publish_s": state.publish.oldest_age(redis, at).await?,
        "publish_set_aside": state.publish.dead_count(redis).await?,
        "publish_lost": state.publish.lost_count(redis).await?,
        "verdicts": accounts["verdicts"],
        "validators": accounts["validators"],
        "current_round": current,
        "embed_tasks": state.settings.embed_tasks,
        "embed_model": state.settings.embed_model,
        "pools": accounts["pools"],
        "coverage": accounts["coverage"],
    })))
}
