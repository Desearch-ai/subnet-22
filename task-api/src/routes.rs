//! Claims, completions and verdicts, the bot's enqueue and room, and the public round pages.

use std::future::Future;
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
use desearch::time::now;
use redis::AsyncCommands;
use serde_json::{json, Map, Value};

use crate::auth::{self, client_address, Caller};
use crate::budgets::{self, CRAWL, EMBED, SHARE_WINDOW_H, WAITING_PER_BUDGET};
use crate::clock::Tick;
use crate::embedding::{self, VECTORS_SUFFIX, VECTORS_TYPE};
use crate::http::{json_body, unavailable, Answer, ApiError, Query, CLAIM_PATH};
use crate::lifecycle::{self, kind_of, signed_manifest, text, EMBED_FIELDS};
use crate::manifest::REVEAL_AFTER_BLOCKS;
use crate::models::{self, ClaimBody, CompleteBody, Enqueue};
use crate::outages;
use crate::py::round_to;
use crate::queues::{self, Claim, Refusal, CLAIMS};
use crate::roundlog::{self, Receipt};
use crate::rounds::UPLOAD_GRACE_S;
use crate::state::State;
use crate::storage::{delete_quietly, is_parquet};
use crate::validations::{self, TaskFilter};
use crate::verdicts::{build_embed_vote, build_vote};
use crate::{journal, roundstore, rules, sampling, uploadlog};

const COMPLETE_LOCK_S: i64 = 300;
const CLAIM_DRAWS: usize = 4;
const COMPLETED_TTL_S: u64 = 900;
const COMPLETE_WAIT: Duration = Duration::from_secs(20);
const COMPLETE_POLL: Duration = Duration::from_millis(250);
/// R2's clock and ours may differ by this much.
const CLOCK_SLACK_S: f64 = 2.0;
const SLOW_S: f64 = 5.0;
const LINK_SLACK_S: u64 = 300;
const TASKS_PAGE: i64 = 50;
const MAX_TASKS_PAGE: i64 = 100;
const ENQUEUED_TTL_S: u64 = 86_400;
/// The sitemap protocol's limit; a longer URL is refused before it can reach a task.
pub const MAX_URL_CHARS: usize = 2048;

/// Seconds each step of a request took, for the log when the whole of it was slow.
struct Steps {
    last: f64,
    done: Vec<(&'static str, f64)>,
}

impl Steps {
    fn since(at: f64) -> Steps {
        Steps { last: at, done: Vec::new() }
    }

    fn mark(&mut self, step: &'static str) {
        let at = now();
        self.done.push((step, at - self.last));
        self.last = at;
    }

    fn slow(&self) -> Option<String> {
        let total: f64 = self.done.iter().map(|(_, seconds)| seconds).sum();
        let steps: Vec<String> = self.done.iter().map(|(step, seconds)| format!("{step} {seconds:.1}s")).collect();
        (total > SLOW_S).then(|| format!("{total:.1}s ({})", steps.join(", ")))
    }
}

/// Runs work a caller may stop waiting for, counted so a stopping copy can say what it cuts short.
fn detach<T: Send + 'static>(state: &Arc<State>, work: impl Future<Output = T> + Send + 'static) -> tokio::task::JoinHandle<T> {
    state.detached.fetch_add(1, Ordering::SeqCst);
    let counted = state.clone();
    tokio::spawn(async move {
        let outcome = work.await;
        counted.detached.fetch_sub(1, Ordering::SeqCst);
        outcome
    })
}

/// What a refused miner should wait, so idle polling does not fill the signed log.
fn retry_after_for(code: &str) -> f64 {
    match code {
        "QUEUE_EMPTY" | "ALREADY_HELD" | "WAITING_FOR_VERDICTS" => 10.0,
        "NO_CAPACITY" => 5.0,
        "VALIDATION_BACKLOG" => 30.0,
        "KIND_CLOSED" | "MODEL_MISMATCH" => 3600.0,
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
        let address = client_address(self.headers, &self.peer.ip().to_string(), &state.settings.client_address_header);
        auth::caller(state, self.headers, self.method, self.uri.path(), self.body, &address).await
    }

    /// The request as the log keeps it, so anyone can check the caller signed it.
    pub fn logged(&self) -> (Value, Vec<journal::Blob>) {
        journal::signed_request(self.headers, self.method, self.uri.path(), self.body)
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
    let mut steps = Steps::since(now());
    let who = signed.caller(state).await?;
    steps.mark("caller");
    if who.is_validator || who.is_admin {
        return Err(ApiError::status(403, "validators may not claim tasks"));
    }
    let body = ClaimBody::parse(&json_body(signed.body)?)?;
    let kind: &'static str = if body.kind == EMBED { EMBED } else { CRAWL };
    let round_id = state.current_round(kind);
    if let Some(pause) = state.custody.paused() {
        return refused(state, signed, &round_id, &who, Refusal::new("PAUSED", json!({"reason": pause["reason"], "since": pause["since"]}))).await;
    }

    let rated = state.retry_after(&who.hotkey).await?;
    steps.mark("rate");
    if let Some(retry_after) = rated {
        // Not logged: signing and storing every excess poll would let a flooder grow the log.
        let refusal = json!({"code": "RATE_LIMITED", "inputs": {"per_sec": state.settings.poll_rate, "retry_after": retry_after}});
        return Ok(Json(json!({"tasks": [], "refusal": refusal, "receipt": null})));
    }
    if kind == EMBED && !state.settings.embed_tasks {
        return refused(state, signed, &round_id, &who, Refusal::new("KIND_CLOSED", json!({"kind": kind}))).await;
    }
    if kind == EMBED && !body.models.contains(&state.settings.embed_model) {
        return refused(state, signed, &round_id, &who, Refusal::new("MODEL_MISMATCH", json!({"model": state.settings.embed_model}))).await;
    }
    let tick = state.tick().await?;
    let at = tick.at;
    let (hotkey, pool) = (who.hotkey.clone(), kind.to_string());
    let locked = state.lookups.read(move |conn| budgets::locked_until(conn, &hotkey, &pool, at)).await?;
    steps.mark("lockout");
    if let Some(until) = locked {
        let clock = state.clock().await?;
        let inputs = json!({"until": round_to(until, 3), "until_block": clock.block_of(until), "retry_after": round_to(until - at, 3)});
        return refused(state, signed, &round_id, &who, Refusal::new("LOCKED_OUT", inputs)).await;
    }
    // Stop leasing work whose upload would expire before validation.
    let validating = state.validation.oldest_age(&state.redis, at).await?.max(state.validation.oldest_seeding_age(&state.redis, at).await?);
    let publishing = state.publish.oldest_age(&state.redis, at).await?;
    let max_backlog = state.settings.max_backlog;
    steps.mark("backlog");
    if max_backlog != 0.0 && validating.max(publishing) > max_backlog {
        let waiting = state.publish.waiting(&state.redis).await?;
        let inputs = json!({"validation_s": validating, "publish_s": publishing, "limit_s": max_backlog, "publish_waiting": waiting});
        return refused(state, signed, &round_id, &who, Refusal::new("VALIDATION_BACKLOG", inputs)).await;
    }
    let (hotkey, pool) = (who.hotkey.clone(), kind.to_string());
    let found = {
        let (hotkey, pool) = (hotkey.clone(), pool.clone());
        state.lookups.read(move |conn| budgets::find(conn, &hotkey, &pool)).await?
    };
    let budget = match found {
        Some(miner) => miner.budget,
        None => {
            let created = state.db.run(move |conn| {
                let miner = budgets::get_or_create(conn, &hotkey, &pool)?;
                // A miner's first claim gives it a budget, whether or not the claim gets anything or is logged.
                if conn.changes() > 0 {
                    journal::append(conn, journal::Record::new("miner", now(), json!({"hotkey": hotkey, "pool": pool}), json!({"budget": miner.budget})))?;
                }
                Ok(miner)
            });
            created.await?.budget
        }
    };
    // The miner's on-chain incentive caps what the checks earned it: validators' own judgment of its pages reaches it this way.
    let budget = budget.min(state.budget_cap(&who.hotkey, at).await?);
    steps.mark("budget");
    let tasks = state.tasks(kind);
    let mut issued = Vec::new();
    let mut receipts = Vec::new();
    let mut dropped = 0;
    // A draw that was all dropped tasks leaves the miner nothing, so it draws again rather than wait out a refusal.
    for _ in 0..CLAIM_DRAWS {
        let claims = match tasks.claim(&state.redis, &who.hotkey, budget, WAITING_PER_BUDGET * budget, body.count, at).await? {
            Ok(claims) => claims,
            Err(refusal) => return refused(state, signed, &round_id, &who, refusal).await,
        };
        steps.mark("queue");
        let drawn = claims.len();
        let mut dropped_now = 0;
        for got in claims {
            if kind == EMBED {
                let handbacks = queues::handbacks(&state.redis, &got.task_id).await?;
                if let Some(cause) = lifecycle::spent(&state.settings, &got.payload, handbacks, now()) {
                    lifecycle::drop_claimed(state, &got.task_id, &got.payload, &who.hotkey, cause, got.seq).await?;
                    dropped_now += 1;
                    continue;
                }
            }
            if let Some(task) = issue(state, &got, kind, &who, at).await? {
                receipts.push(Receipt {
                    round_id: text(&task, "round_id").into(),
                    hotkey: who.hotkey.clone(),
                    requested_at: who.requested_at,
                    outcome: "issued",
                    seq: got.seq,
                    task_id: Some(got.task_id.clone()),
                    request: Some(who.request.clone()),
                    ..Receipt::default()
                });
                issued.push(task);
            }
        }
        dropped += dropped_now;
        if dropped_now < drawn {
            break;
        }
    }
    if issued.is_empty() && dropped > 0 {
        let refusal = Refusal::new("QUEUE_EMPTY", json!({"depth": tasks.depth(&state.redis).await?}));
        return refused(state, signed, &round_id, &who, refusal).await;
    }
    if issued.is_empty() {
        return Err(ApiError::status(503, "upload signing is unavailable"));
    }
    steps.mark("issue");
    let (request, blobs) = signed.logged();
    let input = json!({"request": request, "kind": kind, "budget": budget, "at": at, "block": tick.block});
    let receipts = state.record_logged(receipts, "claim", input, blobs).await?;
    steps.mark("receipts");
    // The time spent signing and recording the claims is ours, not the miner's: the claim starts at the block finalized now.
    let task_ids: Vec<String> = issued.iter().map(|task| text(task, "task_id").to_string()).collect();
    let (clock, started) = (state.clock().await?, state.tick().await?);
    let due = clock.deadline(started, tasks.claim_ttl);
    let expiry = tasks.start_clocks(&state.redis, &who.hotkey, &task_ids, due.at).await?;
    steps.mark("clocks");
    if let Some(slow) = steps.slow() {
        eprintln!("claiming {} tasks for {} took {slow}", task_ids.len(), who.hotkey);
    }
    // Miners are shown the wall time the deadline falls at if blocks keep coming every block time; the block decides.
    let expires_at = now() + (expiry - started.at) - UPLOAD_GRACE_S;
    for task in &mut issued {
        task["expires_at"] = expires_at.into();
        task["expires_block"] = due.block.into();
        task["upload"]["expires_at"] = expires_at.into();
    }
    Ok(Json(json!({"tasks": issued, "receipts": receipts})))
}

async fn refused(state: &State, signed: &Signed<'_>, round_id: &str, who: &Caller, mut refusal: Refusal) -> Answer {
    refusal.inputs.entry("retry_after").or_insert_with(|| retry_after_for(refusal.code).into());
    let refusal_value = refusal.as_value();
    if !state.first_refusal(&who.hotkey, refusal.code).await? {
        return Ok(Json(json!({"tasks": [], "refusal": refusal_value, "receipt": null})));
    }
    let seq = state.next_seq().await?;
    let (request, blobs) = signed.logged();
    let receipt = Receipt {
        round_id: round_id.into(),
        hotkey: who.hotkey.clone(),
        requested_at: who.requested_at,
        outcome: "refused",
        seq,
        refusal: Some(refusal_value.clone()),
        request: Some(who.request.clone()),
        ..Receipt::default()
    };
    let receipt = state.record_logged(vec![receipt], "claim", json!({"request": request}), blobs).await?.remove(0);
    Ok(Json(json!({"tasks": [], "refusal": refusal_value, "receipt": receipt})))
}

/// The task as the miner receives it, with an upload link only it can use.
async fn issue(state: &State, got: &Claim, kind: &str, who: &Caller, at: f64) -> Result<Option<Value>, ApiError> {
    let task_id = &got.task_id;
    let (suffix, content_type) = if kind == EMBED { (VECTORS_SUFFIX, VECTORS_TYPE) } else { (".parquet", PARQUET) };
    // A random name says nothing of the miner, the task or the claim, so neither the storage nor the path to it
    // can tell whose upload a write carries; the key reaches the public log only with the completion. Being
    // random, it also spreads writes over storage that shards by key range.
    let name = format!("{}{suffix}", upload_name());
    let upload_key = format!("uploads/{name}");
    let since_epoch = Duration::from_secs_f64(now());
    // It may outlive the claim; a late completion or a file written after it is refused.
    let claim_ttl = state.settings.claim_ttl_for(kind) as u64;
    let upload_url = state.storage.presign("PUT", &upload_key, claim_ttl + LINK_SLACK_S, Some(content_type), since_epoch);
    let inputs = if kind == EMBED {
        let payload = &got.payload;
        match (payload.get("model"), payload.get("texts"), payload.get("input_key").and_then(Value::as_str), payload.get("input_sha256")) {
            (Some(model), Some(texts), Some(input_key), Some(sha256)) => {
                let url = state.storage.presign("GET", input_key, claim_ttl, None, since_epoch);
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
    let issued = json!({"hotkey": who.hotkey, "key": upload_key, "name": name, "at": at});
    let _: () = state.redis.clone().set(format!("issued:{task_id}"), issued.to_string()).await?;
    let expires_at = now() + (got.expires_at - at) - UPLOAD_GRACE_S;
    let round_id = got.payload.get("round_id").cloned().unwrap_or_else(|| state.current_round(kind).into());
    let mut task = Map::new();
    task.insert("task_id".into(), task_id.clone().into());
    task.insert("kind".into(), kind.into());
    task.insert("round_id".into(), round_id);
    task.insert("expires_at".into(), expires_at.into());
    task.insert("urls".into(), got.payload.get("urls").cloned().unwrap_or_else(|| json!([])));
    task.extend(inputs.as_object().cloned().unwrap_or_default());
    task.insert("upload".into(), json!({"url": upload_url, "key": upload_key, "content_type": content_type, "expires_at": expires_at}));
    Ok(Some(Value::Object(task)))
}

/// 128 random bits from the thread's CSPRNG, as 32 hex characters.
pub fn upload_name() -> String {
    format!("{:032x}", rand::random::<u128>())
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
    // The deadline is judged at the block finalized when the request arrived, not after our own work on it.
    let arrived = Arrival { wall: now(), tick: state.tick().await? };
    let signed = Signed { headers: &headers, method: "POST", uri: &uri, body: &body, peer };
    let who = signed.caller(&state).await?;
    let logged = signed.logged();
    let report = match json_body(&body).and_then(|body| Ok(CompleteBody::parse(&body)?)) {
        Ok(report) => report,
        Err(error) => return refuse_logged(&state, "complete", logged, &task_id, error, arrived.tick, &who.hotkey).await,
    };
    let holds = state.claim_holder(&task_id).await?.as_deref() == Some(who.hotkey.as_str());
    if !holds && completed_by(&state, &task_id, &who.hotkey).await?.is_none() {
        let error = ApiError::status(409, "you do not hold this claim");
        return refuse_logged(&state, "complete", logged, &task_id, error, arrived.tick, &who.hotkey).await;
    }
    let lock = match hold_completion(&state, &task_id, &who.hotkey).await {
        Ok(lock) => lock,
        Err(error) => return refuse_logged(&state, "complete", logged, &task_id, error, arrived.tick, &who.hotkey).await,
    };
    // Detached so a miner that times out can't cancel it and leave the lock to block its retries.
    let finishing = detach(&state.clone(), async move {
        let outcome = finish_completion(&state, &task_id, &report, &who, arrived, logged.clone()).await;
        if let Err(ApiError::Status { status, detail, .. }) = &outcome {
            log_refused(&state, "complete", logged, &task_id, (*status, detail), arrived.tick, None).await?;
        }
        state.delete(&lock).await?;
        outcome
    });
    match finishing.await {
        Ok(outcome) => outcome.map(Json),
        Err(error) => Err(anyhow::anyhow!("completion task failed: {error}").into()),
    }
}

/// A signed request the API turned down, with why, at the rule time it was judged at; a miner holding the claim can then show the
/// log what it was told. A report its task could not have produced counts against its validator in the same transaction.
async fn log_refused(
    state: &State,
    kind: &'static str,
    (request, blobs): (Value, Vec<journal::Blob>),
    task_id: &str,
    (status, detail): (u16, &Value),
    tick: Tick,
    wrong: Option<(String, String)>,
) -> Result<(), ApiError> {
    let at = tick.at;
    let mut input = json!({"request": request, "task_id": task_id, "block": tick.block});
    if let Some((_, why)) = &wrong {
        input["wrong_report"] = why.clone().into();
    }
    let record = journal::Record::new(kind, at, input, json!({"refused": {"status": status, "detail": detail}})).with_blobs(blobs);
    let task = task_id.to_string();
    state
        .db
        .run(move |conn| {
            if let Some((validator, why)) = &wrong {
                rules::note_wrong_report(conn, validator, &task, why, at)?;
            }
            journal::append(conn, record)
        })
        .await?;
    Ok(())
}

/// Refuses a miner's signed completion or abandon and logs the refusal, so every answer the API gives a request a validator
/// relayed is in its log, accepted or not, at the block it was judged at. Past `REFUSALS_LOGGED_PER_MINUTE` refusals of one
/// hotkey in a minute the rest are answered 429 unlogged, so nobody can grow the log at will.
async fn refuse_logged<T>(
    state: &State,
    kind: &'static str,
    logged: (Value, Vec<journal::Blob>),
    task_id: &str,
    error: ApiError,
    tick: Tick,
    hotkey: &str,
) -> Result<T, ApiError> {
    if let ApiError::Status { status, detail, .. } = &error {
        if !state.refusal_to_log(hotkey).await? {
            return Err(ApiError::retry(429, "too many refused requests this minute; they are no longer logged", "60"));
        }
        log_refused(state, kind, logged, task_id, (*status, detail), tick, None).await?;
    }
    Err(error)
}

/// When a completion reached the API: its block and rule time decide whether it was in time, the wall time is compared with storage's own clock.
#[derive(Clone, Copy)]
struct Arrival {
    wall: f64,
    tick: Tick,
}

async fn finish_completion(
    state: &State,
    task_id: &str,
    report: &CompleteBody,
    who: &Caller,
    arrived: Arrival,
    logged: (Value, Vec<journal::Blob>),
) -> Result<Value, ApiError> {
    if let Some(done) = completed_by(state, task_id, &who.hotkey).await? {
        return Ok(done);
    }
    if state.claim_holder(task_id).await?.as_deref() != Some(who.hotkey.as_str()) {
        return Err(ApiError::status(409, "you do not hold this claim"));
    }
    match queues::claim_expiry(&state.redis, task_id).await? {
        Some(expiry) if expiry >= arrived.tick.at => {}
        _ => return Err(ApiError::status(409, "you do not hold a live claim on this task")),
    }
    let result = verify_completion(state, task_id, report, who, arrived, logged).await?;
    let done = json!({"miner": who.hotkey, "result": result});
    let _: () = state.redis.clone().set_ex(format!("completed:{task_id}"), done.to_string(), COMPLETED_TTL_S).await?;
    Ok(result)
}

/// Entity tags compared without the quotes some stores and clients drop.
fn same_etag(a: &str, b: &str) -> bool {
    a.trim().trim_matches('"') == b.trim().trim_matches('"')
}

/// Storage failed a miner's completion: the janitor probes it at once, so an outage is on record before claims lapse in it.
fn storage_down(state: &State, what: &str, error: StorageError) -> ApiError {
    eprintln!("storage failed a completion, probing it now: {what}: {error}");
    state.probe_storage.store(true, std::sync::atomic::Ordering::Release);
    ApiError::status(502, "object storage is unavailable, retry")
}

async fn verify_completion(
    state: &State,
    task_id: &str,
    report: &CompleteBody,
    who: &Caller,
    arrived: Arrival,
    (request, mut blobs): (Value, Vec<journal::Blob>),
) -> Result<Value, ApiError> {
    let issued: Option<String> = state.redis.clone().get(format!("issued:{task_id}")).await?;
    let issued: Value = issued.and_then(|raw| serde_json::from_str(&raw).ok()).unwrap_or_else(|| json!({}));
    if issued.get("key").and_then(Value::as_str) != Some(report.key.as_str()) {
        return Err(ApiError::status(400, "key is not the one issued with this claim"));
    }
    // The version the miner's own upload made, when it names one; otherwise whatever the key holds now.
    let found =
        state.storage.head_at(&report.key, report.version.as_deref()).await.map_err(|e| storage_down(state, &format!("HEAD failed for {}", report.key), e))?;
    let Some(found) = found.filter(|f| f.size > 0) else {
        if report.version.is_some() {
            return Err(ApiError::status(422, "the issued key holds no such version"));
        }
        return Err(ApiError::status(422, "nothing was uploaded to the issued key"));
    };
    if report.etag.as_deref().is_some_and(|etag| !same_etag(etag, &found.etag)) {
        return Err(ApiError::status(409, "the issued key holds another object than the one you uploaded; upload again and complete"));
    }
    if found.modified.is_some_and(|modified| modified as f64 > arrived.wall + CLOCK_SLACK_S) {
        return Err(ApiError::status(409, "the upload was written after this completion"));
    }
    if found.size > state.settings.max_upload {
        delete_quietly(&state.storage, &report.key).await;
        return Err(ApiError::status(413, format!("upload is {} bytes, the limit is {}", found.size, state.settings.max_upload)));
    }
    let payload = state.payload(task_id).await?.unwrap_or_else(|| json!({}));
    let kind = kind_of(&payload).to_string();
    if kind == EMBED {
        let model = text(&payload, "model");
        let model =
            embedding::model(model).ok_or_else(|| anyhow::anyhow!("task={task_id} names {model:?}, a model whose vectors the task API does not know"))?;
        let texts = payload.get("texts").and_then(Value::as_u64).unwrap_or(0);
        let expected = embedding::upload_bytes(model, texts);
        if found.size != expected {
            let detail = format!("the upload is {} bytes; {texts} vectors of {} float16 values are {expected}", found.size, model.dims);
            return Err(ApiError::status(422, detail));
        }
    } else {
        let framed = match is_parquet(&state.storage, &report.key, &found).await {
            Ok(framed) => framed,
            Err(StorageError::Changed | StorageError::Missing) => return Err(ApiError::status(409, "the upload changed while completing, retry")),
            Err(error) => return Err(storage_down(state, &format!("could not read the ends of {}", report.key), error)),
        };
        if !framed {
            return Err(ApiError::status(422, "the upload is not a Parquet file"));
        }
    }
    // Locked for the check window, so not even the bucket's owner can delete the version every reader is sent to. A store
    // that refuses fails the completion closed: nothing unlocked is pinned, the miner completes again, and the refusal opens an
    // `upload_lock` outage, closed by the next lock that holds, so a claim that lapses meanwhile costs its miner nothing.
    let mut lock = Value::Null;
    if let Some(version) = found.version.as_deref().filter(|_| state.settings.upload_lock_s > 0.0) {
        let until = (arrived.wall + state.settings.upload_lock_s).ceil() as i64;
        let tick = arrived.tick;
        if let Err(error) = state.storage.lock_version(&report.key, version, until).await {
            let detail = format!("could not lock {} version {version}: {error}", report.key);
            eprintln!("{detail}");
            state.db.run(move |conn| outages::begin_logged(conn, outages::UPLOAD_LOCK, tick.at, tick.block, &detail)).await?;
            return Err(ApiError::retry(503, "object storage would not lock your upload; complete again", "10"));
        }
        state.db.run(move |conn| outages::end_logged(conn, outages::UPLOAD_LOCK, tick.at, tick.block)).await?;
        lock = json!({"version": version, "until": until, "locked": true});
    }
    state
        .storage
        .make_public_at(&report.key, found.version.as_deref())
        .await
        .map_err(|e| storage_down(state, &format!("could not open {} to readers", report.key), e))?;

    let mut steps = Steps::since(arrived.wall);
    steps.mark("checks");
    let round_id = Some(text(&payload, "round_id")).filter(|r| !r.is_empty()).map(String::from).unwrap_or_else(|| state.current_round(&kind));
    // The sample seed is a block hash nobody knew when the upload was pinned.
    let frozen_block = state.seeds.current_block().await?;
    let completed_at = arrived.tick.at;
    let clock = state.clock().await?;
    let position = payload.get("position").cloned().unwrap_or_else(|| 0.into());
    // The PUT URL outlives this call, so every reader asks for the version checked here: by its version id on a
    // versioned bucket, where a later write can't take its place, else by its entity tag.
    let mut job = json!({
        "task_id": task_id,
        "kind": kind,
        "round_id": round_id,
        "miner": who.hotkey,
        "key": report.key,
        "etag": found.etag,
        "urls": payload.get("urls").cloned().unwrap_or_else(|| json!([])),
        "position": position,
        "rank": payload.get("rank").cloned().unwrap_or(position),
        "attempts": payload.get("attempts").cloned().unwrap_or_else(|| 0.into()),
        "size": found.size,
        "claimed_at": issued.get("at").cloned().unwrap_or(Value::Null),
        "completed_at": completed_at,
        "frozen_block": frozen_block,
        "seed_block": frozen_block + REVEAL_AFTER_BLOCKS,
        "completed_block": arrived.tick.block,
        "deadline": clock.deadline(arrived.tick, state.seeds.wait_s() + state.settings.validation_ttl).at,
    });
    for name in EMBED_FIELDS {
        if let Some(value) = payload.get(name) {
            job[name] = value.clone();
        }
    }
    if let Some(version) = &found.version {
        job["version"] = version.clone().into();
    }
    // An embed upload's counts come from its input, never from the miner.
    if kind == CRAWL {
        job["reported"] = report.reported.clone();
    }
    job["manifest"] = signed_manifest(state, &job);
    steps.mark("manifest");
    let Some(seq) = state.tasks(&kind).complete(&state.redis, task_id, &who.hotkey, &job, &report.key, arrived.tick.at).await? else {
        return Err(ApiError::status(409, "you do not hold a live claim on this task"));
    };
    steps.mark("queues");
    sampling::note_upload(&state.redis, &kind, &who.hotkey, completed_at).await?;
    uploadlog::note(&state.redis, &job).await?;
    steps.mark("logs");
    let (kept, urls) = lifecycle::logged_job(&job);
    blobs.extend(urls);
    let receipt = Receipt {
        round_id,
        hotkey: who.hotkey.clone(),
        requested_at: who.requested_at,
        outcome: "completed",
        seq,
        task_id: Some(task_id.into()),
        block: Some(frozen_block),
        request: Some(who.request.clone()),
        ..Receipt::default()
    };
    let input = json!({"request": request, "task_id": task_id, "job": kept, "lock": lock, "block": arrived.tick.block});
    let receipt = state.record_logged(vec![receipt], "complete", input, blobs).await;
    steps.mark("receipt");
    if let Some(slow) = steps.slow() {
        eprintln!("completing {task_id} took {slow}");
    }
    // The signed receipt goes back with the answer, so a relay that carried the completion holds the API to it.
    let receipt = receipt?.into_iter().next().unwrap_or(Value::Null);
    Ok(json!({"task_id": task_id, "status": "open_for_validation", "receipt": receipt}))
}

pub async fn abandon(
    Shared(state): Shared<Arc<State>>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    Path(task_id): Path<String>,
    headers: HeaderMap,
    uri: Uri,
    body: Bytes,
) -> Answer {
    let signed = Signed { headers: &headers, method: "POST", uri: &uri, body: &body, peer };
    let who = signed.caller(&state).await?;
    let tick = state.tick().await?;
    let at = tick.at;
    let payload = state.payload(&task_id).await?.unwrap_or_else(|| json!({}));
    let kind = kind_of(&payload).to_string();
    let not_held = || ApiError::status(409, "you do not hold this claim");
    if state.claim_holder(&task_id).await?.as_deref() != Some(who.hotkey.as_str()) {
        return refuse_logged(&state, "abandon", signed.logged(), &task_id, not_held(), tick, &who.hotkey).await;
    }
    let expiry = queues::claim_expiry(&state.redis, &task_id).await?;
    let forgiving = lifecycle::forgiving(&state, state.tasks(&kind).claim_ttl).await?;
    // A completion still running decides first; a finished one leaves nothing to abandon.
    let lock = match hold_completion(&state, &task_id, &who.hotkey).await {
        Ok(lock) => lock,
        Err(error) => return refuse_logged(&state, "abandon", signed.logged(), &task_id, error, tick, &who.hotkey).await,
    };
    let seq = state.tasks(&kind).abandon(&state.redis, &task_id, &who.hotkey).await;
    state.delete(&lock).await?;
    let Some(seq) = seq? else {
        return refuse_logged(&state, "abandon", signed.logged(), &task_id, not_held(), tick, &who.hotkey).await;
    };
    let round_id = Some(text(&payload, "round_id")).filter(|r| !r.is_empty()).map(String::from).unwrap_or_else(|| state.current_round(&kind));
    let receipt = Receipt {
        round_id,
        hotkey: who.hotkey.clone(),
        requested_at: who.requested_at,
        outcome: "reclaimed",
        seq,
        task_id: Some(task_id.clone()),
        request: Some(who.request.clone()),
        ..Receipt::default()
    };
    let (request, blobs) = signed.logged();
    let (hotkey, task, lapsed, key) = (who.hotkey.clone(), task_id.clone(), payload, state.key.clone());
    let (budget, receipts) = state
        .db
        .run(move |conn| {
            // Decided here, on the outages the log holds up to this entry, so a replay decides it alike.
            let forgiven = lifecycle::forgiven_in(conn, &forgiving, expiry)?;
            let receipt = Receipt { cause: Some(forgiven.clone().unwrap_or_else(|| "abandoned".into())), ..receipt };
            let receipts = crate::state::record_receipts(conn, &key, vec![receipt], at)?;
            let budget = if forgiven.is_some() {
                budgets::get_or_create(conn, &hotkey, kind_of(&lapsed))?.budget
            } else {
                lifecycle::lapse(conn, &hotkey, &task, &lapsed, "abandoned", at)?
            };
            let input = json!({
                "request": request,
                "task_id": task,
                "kind": kind_of(&lapsed),
                "assigned": lifecycle::assigned(&lapsed),
                "expiry": expiry,
                "forgiving": forgiving.logged(),
                "forgiven": forgiven,
                "block": tick.block,
            });
            journal::append(conn, journal::Record::new("abandon", at, input, json!({"receipts": receipts.clone(), "budget": budget})).with_blobs(blobs))?;
            Ok((budget, receipts))
        })
        .await?;
    Ok(Json(json!({"task_id": task_id, "budget": budget, "receipts": receipts})))
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
    let signed = Signed { headers: &headers, method: "POST", uri: &uri, body: &body, peer };
    let who = validator(&state, &signed).await?;
    models::release(&json_body(&body)?)?;
    let job = state.validation.job(&state.redis, &task_id).await?;
    let Some(job) = job.filter(|job| job.get("picked").is_some_and(|p| !p.is_null() && p != false && p != "")) else {
        return Err(ApiError::status(409, "no such open upload"));
    };
    let pinned = job.get("version").and_then(Value::as_str);
    let gone = state.storage.head_at(text(&job, "key"), pinned).await.map_err(|_| ApiError::status(502, "object storage is unavailable, retry"))?.is_none();
    if !gone {
        return Ok(Json(json!({"task_id": task_id, "status": "open"})));
    }
    let Some(lapsed) = state.validation.finalize(&state.redis, &task_id, None, false, now()).await? else {
        return Err(ApiError::status(409, "the upload was finalized"));
    };
    lifecycle::void_task(&state, &task_id, lapsed, &who.hotkey, "upload_missing").await?;
    let (request, blobs) = signed.logged();
    let record = journal::Record::new("release", now(), json!({"request": request, "task_id": task_id}), json!({"status": "void", "reason": "upload_missing"}));
    let record = record.with_blobs(blobs);
    state.db.run(move |conn| journal::append(conn, record)).await?;
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
    let signed = Signed { headers: &headers, method: "POST", uri: &uri, body: &body, peer };
    let who = validator(&state, &signed).await?;
    let result = match json_body(&body)? {
        Value::Object(result) => result,
        Value::Null => return Err(models::Invalid::single("missing", &["body"], "Field required").into()),
        _ => return Err(models::Invalid::single("dict_type", &["body"], "Input should be a valid dictionary").into()),
    };
    let lock = lifecycle::lock_key(&task_id);
    if !state.set_once(&lock, &who.hotkey, lifecycle::FINALIZE_LOCK_S).await? {
        return Err(ApiError::retry(503, "a verdict for this upload is already being recorded", "2"));
    }
    let tick = state.tick().await?;
    let at = tick.at;
    let mut wrong = None;
    let outcome = record_verdict(&state, &task_id, result, &who, signed.logged(), at, &mut wrong).await;
    if let Err(ApiError::Status { status, detail, .. }) = &outcome {
        log_refused(&state, "report", signed.logged(), &task_id, (*status, detail), tick, wrong).await?;
    }
    state.delete(&lock).await?;
    outcome.map(Json)
}

async fn record_verdict(
    state: &State,
    task_id: &str,
    result: Map<String, Value>,
    who: &Caller,
    (request, blobs): (Value, Vec<journal::Blob>),
    at: f64,
    wrong: &mut Option<(String, String)>,
) -> Result<Value, ApiError> {
    let (hotkey, since, limit) = (who.hotkey.clone(), at - state.settings.wrong_reports_window_s, state.settings.wrong_reports);
    if state.lookups.read(move |conn| validations::is_excluded(conn, &hotkey, since, limit)).await? {
        state.validation.leave(&state.redis, &who.hotkey).await?;
        return Err(ApiError::status(403, format!("this validator sent {limit} or more reports its tasks could not have produced in the last day")));
    }
    state.validation.present(&state.redis, &who.hotkey, at).await?;
    let Some(job) = state.validation.job(&state.redis, task_id).await? else {
        return Err(ApiError::status(409, "no such open upload"));
    };
    let kind = kind_of(&job).to_string();
    state.validation.present_for(&state.redis, &kind, &who.hotkey, at).await?;
    let built =
        if kind == EMBED { build_embed_vote(&job, &who.hotkey, models::embed_score(&result)?) } else { build_vote(&job, &who.hotkey, models::score(&result)?) };
    let mut vote = match built {
        Ok(vote) => vote,
        Err(why) => {
            // A report the task could not have produced is objectively wrong, whoever else agrees with it; it is counted with
            // the refusal's log entry.
            *wrong = Some((who.hotkey.clone(), why.0.clone()));
            return Err(ApiError::status(422, why.0));
        }
    };
    vote["at"] = at.into();
    if state.validation.vote(&state.redis, task_id, &who.hotkey, &vote, at).await? == 0 {
        return Err(ApiError::status(409, "this validator already voted on this upload"));
    }
    let record = journal::Record::new("report", at, json!({"request": request, "task_id": task_id}), json!({"vote": vote})).with_blobs(blobs);
    state.db.run(move |conn| journal::append(conn, record)).await?;
    if let Some(finalized) = lifecycle::finalize_task(state, task_id, &job, at).await? {
        return Ok(finalized);
    }
    let (miner, pool) = (text(&job, "miner").to_string(), kind.clone());
    let budget = state.lookups.read(move |conn| budgets::get(conn, &miner, &pool)).await?.budget;
    Ok(json!({"task_id": task_id, "verdict": "pending", "credited": 0, "miner_budget": budget}))
}

pub async fn shares(Shared(state): Shared<Arc<State>>) -> Answer {
    let as_of = state.rule_now().await? - state.settings.ledger_delay;
    let pools = state.lookups.read(move |conn| budgets::shares(conn, SHARE_WINDOW_H, as_of)).await?;
    Ok(Json(json!({"window_hours": SHARE_WINDOW_H, "as_of": as_of, "pools": pools})))
}

/// How many tasks the queue can take now: crawl tasks for the bot, or with `?kind=embed` embed inputs for the planner.
pub async fn room(Shared(state): Shared<Arc<State>>, QueryParams(query): QueryParams<Query>) -> Answer {
    if query.get("kind").map(String::as_str) == Some(EMBED) {
        return embed_room(&state).await;
    }
    let at = now();
    let depth = state.crawl.depth(&state.redis).await?;
    let unrevealed = state.lookups.read(|conn| roundstore::unrevealed_tasks(conn, CRAWL)).await?;
    let claimed: i64 = state.redis.clone().zcard(CLAIMS).await?;
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
    let room = (state.settings.queue_target - depth - unrevealed).max(0);
    let per_minute = state.publish_rate.lock().expect("publish rate").per_second() * 60.0;
    Ok(Json(json!({
        "room_tasks": if refusing { 0 } else { room },
        "queue": depth,
        "unrevealed": unrevealed,
        "in_system": in_system,
        "published_per_min": round_to(per_minute, 1),
        "refusing": refusing,
    })))
}

/// Embed inputs the planner may hand over now: none while embed tasks are closed or old work holds claims back, and less while validators owe verdicts.
async fn embed_room(state: &State) -> Answer {
    let at = now();
    let depth = state.embed.depth(&state.redis).await?;
    let unrevealed = state.lookups.read(|conn| roundstore::unrevealed_tasks(conn, EMBED)).await?;
    let waiting: i64 = state.redis.clone().llen(queues::EMBED_INPUTS).await?;
    let awaiting = state.validation.open_embed(&state.redis).await?;
    let oldest = state.validation.oldest_age(&state.redis, at).await?.max(state.publish.oldest_age(&state.redis, at).await?);
    let refusing = state.settings.max_backlog != 0.0 && oldest > state.settings.max_backlog;
    let room = if state.settings.embed_tasks && !refusing { (state.settings.embed_queue_target - depth - unrevealed - waiting - awaiting).max(0) } else { 0 };
    Ok(Json(json!({
        "room_tasks": room,
        "queue": depth,
        "unrevealed": unrevealed,
        "waiting_inputs": waiting,
        "awaiting_verdict": awaiting,
        "open": state.settings.embed_tasks,
        "refusing": refusing,
    })))
}

/// A batch sent twice, after a timeout, is queued once.
pub async fn enqueue(Shared(state): Shared<Arc<State>>, ConnectInfo(peer): ConnectInfo<SocketAddr>, headers: HeaderMap, uri: Uri, body: Bytes) -> Answer {
    let signed = Signed { headers: &headers, method: "POST", uri: &uri, body: &body, peer };
    let who = signed.caller(&state).await?;
    if !who.is_admin {
        return Err(ApiError::status(403, "only admins may do this"));
    }
    let request = Enqueue::parse(&json_body(&body)?)?;
    if let Some(long) = request.urls.iter().find(|url| url.url.chars().count() > MAX_URL_CHARS) {
        let start: String = long.url.chars().take(80).collect();
        return Err(ApiError::status(422, format!("a URL longer than {MAX_URL_CHARS} characters: {start}...")));
    }
    if !crate::janitor::stages_rounds(&state).await? {
        return Err(ApiError::retry(503, "an earlier version still leads; send the batch again shortly", "15"));
    }
    let seen = (!request.batch_id.is_empty()).then(|| format!("enqueued:{}", request.batch_id));
    if let Some(seen) = &seen {
        let earlier: Option<String> = state.redis.clone().get(seen).await?;
        if let Some(earlier) = earlier.and_then(|raw| serde_json::from_str::<Value>(&raw).ok()) {
            return Ok(Json(earlier));
        }
    }
    let round = lifecycle::open_round(&state, request.urls, signed.logged()).await?.map_err(|invalid| ApiError::status(422, invalid.to_string()))?;
    let found = json!({"round_id": round.round_id, "batches": round.tasks(), "seed_block": round.seed_block, "manifest_hash": round.manifest_hash});
    if let Some(seen) = &seen {
        let _: () = state.redis.clone().set_ex(seen, found.to_string(), ENQUEUED_TTL_S).await?;
    }
    Ok(Json(found))
}

/// The finalized height every deadline is measured on, and how it turns into the rule times the API stores and shows.
pub async fn clock(Shared(state): Shared<Arc<State>>) -> Answer {
    let tick = state.tick().await?;
    Ok(Json(state.clock().await?.view(tick, now())))
}

pub async fn key(Shared(state): Shared<Arc<State>>) -> Answer {
    Ok(Json(json!({"signer": state.signer()})))
}

/// The storage canary's probes since `since` (the last hour unless asked), newest first, and the outages they show.
pub async fn storage_probes(Shared(state): Shared<Arc<State>>, QueryParams(query): QueryParams<Query>) -> Answer {
    let since = crate::logs::float_param(&query, "since")?.unwrap_or_else(|| now() - 3600.0);
    let limit = crate::logs::int_param(&query, "limit", 100, 1, 1000)?;
    let every = state.settings.canary_every;
    let mut report = state.lookups.read(move |conn| crate::canary::report(conn, since, limit)).await?;
    report["every_s"] = every.into();
    Ok(Json(report))
}

/// Seconds a validator's storage probe link lasts, and links one validator gets a minute.
const PROBE_LINK_S: u64 = 120;
const PROBE_LINKS_PER_MINUTE: i64 = 4;
/// Storage witnesses one validator may log a minute; one is enough for the rules, the rest are answered 429.
const WITNESSES_PER_MINUTE: i64 = 2;

/// A presigned PUT, GET and DELETE of a fresh key named and shaped like a miner's upload, so a validator probes storage the
/// way miners use it, from where it stands; storage can't tell the probe from an upload.
pub async fn storage_probe_link(
    Shared(state): Shared<Arc<State>>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    uri: Uri,
    body: Bytes,
) -> Answer {
    let signed = Signed { headers: &headers, method: "POST", uri: &uri, body: &body, peer };
    let who = validator(&state, &signed).await?;
    if !state.per_minute("probe_links", &who.hotkey, PROBE_LINKS_PER_MINUTE).await? {
        return Err(ApiError::retry(429, "too many probe links this minute", "60"));
    }
    let key = format!("uploads/{}.parquet", upload_name());
    let at = now();
    let since = Duration::from_secs_f64(at);
    Ok(Json(json!({
        "key": key,
        "content_type": PARQUET,
        "put_url": state.storage.presign("PUT", &key, PROBE_LINK_S, Some(PARQUET), since),
        "get_url": state.storage.presign("GET", &key, PROBE_LINK_S, None, since),
        "delete_url": state.storage.presign("DELETE", &key, PROBE_LINK_S, None, since),
        "expires_at": at + PROBE_LINK_S as f64,
    })))
}

/// What a validator's witness says failed, or None when it reports storage working: its own probe (`probe`, `ok` false) or its
/// look at a completed upload (`storage`) that storage did not answer or answered with a server error.
fn witnessed_failure(record: &Value) -> Option<String> {
    let failed = match record["kind"].as_str() {
        Some("probe") => record["ok"] == false,
        Some("storage") => record["status"].as_i64().is_none_or(|status| status >= 500),
        _ => false,
    };
    if !failed {
        return None;
    }
    let what = match record["kind"].as_str() {
        Some("probe") => format!("probe {}: {}", text(record, "step"), text(record, "detail")),
        _ => format!("HEAD {}: {}", text(record, "key"), record["status"].as_i64().map_or_else(|| text(record, "error").to_string(), |s| format!("HTTP {s}"))),
    };
    Some(what.chars().take(200).collect())
}

/// A validator's signed report that storage failed it, logged at the block it arrived at. Wherever enough validators report
/// failures close together, a claim alive then lapses at no cost, whatever the API's own probes saw (`outages::witnessed`).
/// A refused report is logged too, so the validator's own record of sending it finds an answer in the log.
pub async fn storage_witness(
    Shared(state): Shared<Arc<State>>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    uri: Uri,
    body: Bytes,
) -> Answer {
    let signed = Signed { headers: &headers, method: "POST", uri: &uri, body: &body, peer };
    let who = validator(&state, &signed).await?;
    let tick = state.tick().await?;
    let record = json_body(&body)?.get("record").cloned().unwrap_or(Value::Null);
    let failure = witnessed_failure(&record).filter(|_| record["witness"] == who.hotkey.as_str());
    let Some(detail) = failure else {
        let error = ApiError::status(422, "a storage witness must be the caller's own and report a failure");
        return refuse_logged(&state, "witness", signed.logged(), "", error, tick, &who.hotkey).await;
    };
    if !state.per_minute("witnesses", &who.hotkey, WITNESSES_PER_MINUTE).await? {
        return Err(ApiError::retry(429, "too many storage witnesses this minute", "60"));
    }
    let (request, blobs) = signed.logged();
    let validator = who.hotkey.clone();
    let input = json!({"request": request, "witness": validator, "kind": record["kind"], "detail": detail, "at": tick.at, "block": tick.block});
    let seq = state
        .db
        .run(move |conn| {
            outages::note_witness(conn, &validator, tick.at, tick.block, &detail)?;
            let (seq, _) = journal::append(conn, journal::Record::new("witness", tick.at, input, Value::Null).with_blobs(blobs))?;
            Ok(seq)
        })
        .await?;
    Ok(Json(json!({"accepted": true, "seq": seq, "block": tick.block})))
}

/// Every setting this copy runs with, secrets left out; the attestation binds its hash.
pub async fn config(Shared(state): Shared<Arc<State>>) -> Answer {
    Ok(Json(state.settings.public()))
}

/// Which key signs the log, every custody record this copy knows, and the pause.
pub async fn custody(Shared(state): Shared<Arc<State>>) -> Answer {
    Ok(Json(crate::custody::view(&state).await?))
}

/// A predecessor's state, sealed to this CVM's attested handover key.
pub async fn custody_receive(Shared(state): Shared<Arc<State>>, body: Bytes) -> Answer {
    Ok(Json(crate::custody::receive(&state, body).await?))
}

async fn admin(state: &State, signed: &Signed<'_>) -> Result<Caller, ApiError> {
    let who = signed.caller(state).await?;
    if !who.is_admin {
        return Err(ApiError::status(403, "only an admin hotkey may do this"));
    }
    Ok(who)
}

/// Hands the log to the CVM at `successor`, once its attestation holds and validators' stake approved its version.
pub async fn handover(Shared(state): Shared<Arc<State>>, ConnectInfo(peer): ConnectInfo<SocketAddr>, headers: HeaderMap, uri: Uri, body: Bytes) -> Answer {
    admin(&state, &Signed { headers: &headers, method: "POST", uri: &uri, body: &body, peer }).await?;
    let request = json_body(&body)?;
    let successor =
        request["successor"].as_str().filter(|s| s.starts_with("http")).ok_or_else(|| ApiError::status(422, "successor must be the successor's URL"))?;
    Ok(Json(crate::custody::hand_over(&state, successor).await?))
}

/// Stops or resumes handing out tasks, the only emergency power.
pub async fn pause(Shared(state): Shared<Arc<State>>, ConnectInfo(peer): ConnectInfo<SocketAddr>, headers: HeaderMap, uri: Uri, body: Bytes) -> Answer {
    let signed = Signed { headers: &headers, method: "POST", uri: &uri, body: &body, peer };
    let who = admin(&state, &signed).await?;
    let request = json_body(&body)?;
    let paused = request["paused"].as_bool().ok_or_else(|| ApiError::status(422, "paused must be true or false"))?;
    let reason = request["reason"].as_str().unwrap_or_default();
    if paused && reason.trim().is_empty() {
        return Err(ApiError::status(422, "a pause needs a reason"));
    }
    let answer = crate::custody::pause(&state, paused, reason, &who.hotkey).await?;
    // Claims are refused while paused, so the log says when and why, with the admin's signed request.
    let (logged, blobs) = signed.logged();
    let record = journal::Record::new("pause", now(), json!({"request": logged, "paused": paused, "reason": reason}), answer.clone()).with_blobs(blobs);
    state.db.run(move |conn| journal::append(conn, record)).await?;
    Ok(Json(answer))
}

/// The CVM's TDX quote and what it binds: the signer, the handover key, the settings and the TLS key.
pub async fn attestation(Shared(state): Shared<Arc<State>>, QueryParams(query): QueryParams<Query>) -> Answer {
    let Some(tee) = &state.tee else {
        return Err(ApiError::status(404, "this copy does not run in a TEE"));
    };
    let nonce = query.get("nonce").map(|text| crate::attestation::nonce(text)).transpose().map_err(|error| ApiError::status(422, error.to_string()))?;
    let answer = tee.attestation(&state.signer(), &state.settings.public(), nonce.as_deref()).await.map_err(|error| {
        eprintln!("attesting failed: {error:#}");
        ApiError::retry(503, "the attestation could not be made now, retry", "5")
    })?;
    Ok(Json(answer))
}

pub async fn round_list(Shared(state): Shared<Arc<State>>, QueryParams(query): QueryParams<Query>) -> Answer {
    let limit = crate::logs::int_param(&query, "limit", 100, 1, 1000)?;
    let rounds = state.lookups.read(move |conn| roundstore::recent(conn, limit)).await?;
    Ok(Json(json!({"rounds": rounds})))
}

pub async fn round_view(Shared(state): Shared<Arc<State>>, Path(round_id): Path<String>) -> Answer {
    let params = state.settings.rules();
    let Some((found, loads)) = state
        .lookups
        .read(move |conn| {
            let Some(found) = roundstore::get(conn, &round_id)? else { return Ok(None) };
            let loads = rules::round_view(conn, &round_id, &params)?;
            Ok(Some((found, loads)))
        })
        .await?
    else {
        return Err(ApiError::status(404, "no such round"));
    };
    let mut view = found.public_view();
    view.insert("signer".into(), state.signer().into());
    // How much of the round validators could load, and the hosts most miners could not, as its checks so far showed.
    if found.kind == CRAWL {
        view.insert("loadability".into(), loads.view());
    }
    Ok(Json(Value::Object(view)))
}

/// Storage outages and finality stalls the API saw in the last week: claims alive in them lapsed at no cost.
pub async fn outages(Shared(state): Shared<Arc<State>>) -> Answer {
    let at = state.rule_now().await?;
    let since = at - crate::outages::KEEP_DAYS * 86_400.0;
    let (window, quorum) = (state.settings.storage_witness_window, state.settings.storage_witness_quorum);
    let (found, witnessed) =
        state.lookups.read(move |conn| Ok((crate::outages::recent(conn, since, 200)?, crate::outages::witnessed(conn, since, at, window, quorum)?))).await?;
    // Storage outages validators' witnesses make, beside the ones the API's own probes opened.
    let witnessed: Vec<Value> = witnessed
        .iter()
        .map(|span| {
            json!({"cause": crate::outages::WITNESSED, "start": span.start, "end": span.end, "start_block": span.start_block,
                   "end_block": span.end_block, "validators": span.validators})
        })
        .collect();
    Ok(Json(json!({"outages": found, "witnessed": witnessed, "witness_quorum": quorum, "witness_window_s": window})))
}

pub async fn round_log(Shared(state): Shared<Arc<State>>, Path(round_id): Path<String>) -> Answer {
    let asked = round_id.clone();
    let (entries, anchor) = state.lookups.read(move |conn| Ok((roundlog::entries(conn, &asked)?, roundlog::anchored_root(conn, &asked)?))).await?;
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
    let tasks = state.lookups.read(move |conn| validations::recent(conn, &filter)).await?;
    Ok(Json(json!({"tasks": tasks})))
}

pub async fn ping(Shared(state): Shared<Arc<State>>) -> Answer {
    if state.draining.load(Ordering::Relaxed) {
        return Err(ApiError::status(503, "shutting down"));
    }
    // Healthy only once the subnet's hotkeys are known, so a deploy never routes callers to a copy that must turn them away.
    if !state.registry.ready() {
        return Err(ApiError::status(503, "the subnet's hotkeys are still loading"));
    }
    Ok(Json(json!({"ok": true})))
}

pub async fn health(Shared(state): Shared<Arc<State>>) -> Answer {
    let at = now();
    let redis = &state.redis;
    let outcomes: Option<i64> = redis.clone().get(desearch::feeds::OUTCOMES.counter()).await?;
    let rule_now = state.rule_now().await?;
    let mut active = state.validation.active(redis, rule_now).await?;
    active.sort();
    let wrong = (rule_now - state.settings.wrong_reports_window_s, state.settings.wrong_reports);
    let accounts = state
        .lookups
        .read(move |conn| {
            Ok(json!({
                "verdicts": validations::verdicts(conn, None)?,
                "validators": validations::audit_standing(conn, wrong.0, wrong.1)?,
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
        "data_disk_used": crate::db::disk_used(&state.settings.data).map(|used| round_to(used, 3)),
        "verdicts": accounts["verdicts"],
        "validators": accounts["validators"],
        "current_round": current,
        "embed_tasks": state.settings.embed_tasks,
        "embed_model": state.settings.embed_model,
        "pools": accounts["pools"],
        "coverage": accounts["coverage"],
    })))
}
