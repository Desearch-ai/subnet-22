//! The HTTP surface: routes, the body limit, CORS and request rates, and errors answered as FastAPI answered them.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Instant;

use axum::body::{to_bytes, Body};
use axum::extract::{ConnectInfo, DefaultBodyLimit, MatchedPath, Request, State as Shared};
use axum::http::header::{CONNECTION, CONTENT_LENGTH, RETRY_AFTER};
use axum::http::{HeaderName, HeaderValue, Method, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use tower_http::cors::{AllowOrigin, Any, CorsLayer};

use crate::db::Busy;
use crate::models::Invalid;
use crate::state::State;
use crate::{logs, routes};

const BODY_BYTES: usize = 64_000;
const LARGE_BODY_BYTES: usize = 16_000_000;
const UNAVAILABLE_RETRY_S: i64 = 2;
pub const CLAIM_PATH: &str = "/v1/tasks/claim";
/// Requests that spent this long inside the API are counted in the minute's log line.
const SLOW_INSIDE_S: f64 = 5.0;
/// Requests signed this long before they arrived are counted too: they were slow on the way here.
const LATE_ARRIVAL_S: f64 = 10.0;

/// The key a read was signed with, when it was answered, and the signature; `signed_read` says what it covers.
pub const SIGNER_HEADER: &str = "x-desearch-signer";
pub const SIGNED_AT_HEADER: &str = "x-desearch-signed-at";
pub const SIGNATURE_HEADER: &str = "x-desearch-signature";

pub enum ApiError {
    Status {
        status: u16,
        detail: Value,
        retry_after: Option<String>,
    },
    /// Redis is unreachable, restarting or not yet promoted: callers are told to come back.
    Unavailable,
    Internal(anyhow::Error),
}

impl ApiError {
    pub fn status(status: u16, detail: impl Into<String>) -> Self {
        ApiError::Status { status, detail: Value::String(detail.into()), retry_after: None }
    }

    pub fn retry(status: u16, detail: impl Into<String>, retry_after: &str) -> Self {
        ApiError::Status { status, detail: Value::String(detail.into()), retry_after: Some(retry_after.into()) }
    }
}

/// Redis gone, timing out, or a replica that may not be written.
pub fn redis_down(error: &redis::RedisError) -> bool {
    error.is_io_error() || error.is_connection_dropped() || error.is_connection_refusal() || error.is_timeout() || error.code() == Some("READONLY")
}

impl From<anyhow::Error> for ApiError {
    fn from(error: anyhow::Error) -> Self {
        for cause in error.chain() {
            if cause.downcast_ref::<redis::RedisError>().is_some_and(redis_down) {
                return ApiError::Unavailable;
            }
            if cause.is::<Busy>() {
                return ApiError::retry(503, "the database readers are busy, retry", "1");
            }
        }
        ApiError::Internal(error)
    }
}

impl From<redis::RedisError> for ApiError {
    fn from(error: redis::RedisError) -> Self {
        anyhow::Error::from(error).into()
    }
}

impl From<Invalid> for ApiError {
    fn from(invalid: Invalid) -> Self {
        ApiError::Status { status: 422, detail: Value::Array(invalid.0), retry_after: None }
    }
}

fn detail(status: StatusCode, detail: Value) -> Response {
    (status, Json(json!({"detail": detail}))).into_response()
}

fn with_retry_after(mut response: Response, seconds: &str) -> Response {
    if let Ok(value) = HeaderValue::from_str(seconds) {
        response.headers_mut().insert(RETRY_AFTER, value);
    }
    response
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        match self {
            ApiError::Status { status, detail: body, retry_after } => {
                let response = detail(StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), body);
                match retry_after {
                    Some(seconds) => with_retry_after(response, &seconds),
                    None => response,
                }
            }
            ApiError::Unavailable => unavailable(""),
            ApiError::Internal(error) => {
                eprintln!("request failed: {error:#}");
                detail(StatusCode::INTERNAL_SERVER_ERROR, "Internal Server Error".into())
            }
        }
    }
}

/// A claim finds no tasks; anything else is asked to retry.
pub fn unavailable(path: &str) -> Response {
    if path == CLAIM_PATH {
        let refusal = json!({"code": "UNAVAILABLE", "inputs": {"retry_after": UNAVAILABLE_RETRY_S}});
        return Json(json!({"tasks": [], "refusal": refusal, "receipt": null})).into_response();
    }
    with_retry_after(detail(StatusCode::SERVICE_UNAVAILABLE, "temporarily unavailable, retry".into()), &UNAVAILABLE_RETRY_S.to_string())
}

pub type Answer = Result<Json<Value>, ApiError>;

/// A request body as JSON: absent is null, anything unparseable is FastAPI's 422.
pub fn json_body(body: &[u8]) -> Result<Value, ApiError> {
    if body.is_empty() {
        return Ok(Value::Null);
    }
    serde_json::from_slice(body).map_err(|error| Invalid::single("json_invalid", &["body"], &format!("JSON decode error: {error}")).into())
}

/// Query parameters, the last of each name winning.
pub type Query = HashMap<String, String>;

pub fn body_limit(path: &str) -> usize {
    if path == crate::custody::RECEIVE_PATH {
        crate::custody::HANDOVER_BYTES
    } else if path == "/v1/admin/enqueue" || path.ends_with("/score") {
        LARGE_BODY_BYTES
    } else {
        BODY_BYTES
    }
}

fn too_large(limit: usize) -> Response {
    let mut response = detail(StatusCode::PAYLOAD_TOO_LARGE, format!("the body is over {limit} bytes").into());
    response.headers_mut().insert(CONNECTION, HeaderValue::from_static("close"));
    response
}

/// Refuses a request body over its path's limit before the rest of it is read.
async fn limit_body(request: Request, next: Next) -> Response {
    let limit = body_limit(request.uri().path());
    let declared = request.headers().get(CONTENT_LENGTH).and_then(|v| v.to_str().ok()).and_then(|v| v.parse::<usize>().ok());
    if declared.is_some_and(|declared| declared > limit) {
        return too_large(limit);
    }
    let (parts, body) = request.into_parts();
    match to_bytes(body, limit).await {
        Ok(bytes) => next.run(Request::from_parts(parts, Body::from(bytes))).await,
        Err(_) => too_large(limit),
    }
}

pub fn peer(request: &Request) -> String {
    request.extensions().get::<ConnectInfo<SocketAddr>>().map(|ConnectInfo(addr)| addr.ip().to_string()).unwrap_or_default()
}

async fn limit_requests(Shared(state): Shared<Arc<State>>, request: Request, next: Next) -> Response {
    let path = request.uri().path().to_string();
    let address = crate::auth::client_address(request.headers(), &peer(&request), &state.settings.client_address_header);
    let wait = match *request.method() {
        Method::GET => state.read_wait(&address, logs::PATHS.iter().any(|p| path.starts_with(p))).await,
        Method::POST => state.write_wait(&address).await,
        _ => Ok(None),
    };
    match wait {
        Err(error) => match ApiError::from(error) {
            ApiError::Unavailable => unavailable(&path),
            other => other.into_response(),
        },
        Ok(Some(wait)) => with_retry_after(detail(StatusCode::TOO_MANY_REQUESTS, "too many requests".into()), &wait.to_string()),
        Ok(None) => next.run(request).await,
    }
}

/// What a read's signature covers, as canonical JSON: the path and query asked, when it was answered, and the SHA-256 of the body sent.
pub fn signed_read(path: &str, at: &str, body: &[u8]) -> Vec<u8> {
    crate::py::canonical(&json!({"path": path, "at": at, "sha256": crate::proofs::sha256(body)}))
}

/// Every answered read is signed, so a page or a screenshot of one can be shown to have come from the API, and when.
async fn sign_reads(Shared(state): Shared<Arc<State>>, request: Request, next: Next) -> Response {
    if request.method() != Method::GET {
        return next.run(request).await;
    }
    let path = request.uri().path_and_query().map(|p| p.as_str().to_string()).unwrap_or_default();
    let response = next.run(request).await;
    if !response.status().is_success() {
        return response;
    }
    let (mut parts, body) = response.into_parts();
    let Ok(bytes) = to_bytes(body, usize::MAX).await else {
        return detail(StatusCode::INTERNAL_SERVER_ERROR, "Internal Server Error".into());
    };
    let at = format!("{:.3}", desearch::time::now());
    let signature = state.sign(&signed_read(&path, &at, &bytes));
    for (name, value) in [(SIGNER_HEADER, state.signer()), (SIGNED_AT_HEADER, at), (SIGNATURE_HEADER, signature)] {
        if let Ok(value) = HeaderValue::from_str(&value) {
            parts.headers.insert(HeaderName::from_static(name), value);
        }
    }
    Response::from_parts(parts, Body::from(bytes))
}

/// A copy that does not hold the log turns writes away: it would sign with a key validators do not follow.
async fn hold_writes(Shared(state): Shared<Arc<State>>, request: Request, next: Next) -> Response {
    let path = request.uri().path();
    let exempt = path == crate::custody::RECEIVE_PATH || path == crate::custody::HANDOVER_PATH;
    if *request.method() == Method::POST && !exempt && !state.custody.signing() {
        if path == CLAIM_PATH {
            return unavailable(CLAIM_PATH);
        }
        let status = serde_json::to_value(state.custody.status()).unwrap_or_default();
        return with_retry_after(
            detail(StatusCode::SERVICE_UNAVAILABLE, format!("this copy does not hold the log ({})", status.as_str().unwrap_or("")).into()),
            "30",
        );
    }
    next.run(request).await
}

fn cors(origins: &[String]) -> CorsLayer {
    let allowed: Vec<HeaderValue> = origins.iter().filter_map(|o| HeaderValue::from_str(o).ok()).collect();
    CorsLayer::new().allow_origin(AllowOrigin::list(allowed)).allow_methods([Method::GET]).allow_headers(Any).expose_headers([
        HeaderName::from_static("retry-after"),
        HeaderName::from_static(SIGNER_HEADER),
        HeaderName::from_static(SIGNED_AT_HEADER),
        HeaderName::from_static(SIGNATURE_HEADER),
    ])
}

async fn not_found() -> Response {
    detail(StatusCode::NOT_FOUND, "Not Found".into())
}

#[derive(Default)]
struct Minute {
    at: i64,
    answered: u64,
    slow: HashMap<String, (u64, f64)>,
    late: (u64, f64),
}

static MINUTE: LazyLock<Mutex<Minute>> = LazyLock::new(Mutex::default);

/// One log line a minute when requests were slow inside the API or on their way to it.
async fn time_requests(request: Request, next: Next) -> Response {
    let started = Instant::now();
    let arrived = desearch::time::now();
    let route = request.extensions().get::<MatchedPath>().map_or_else(|| "unrouted".to_string(), |path| path.as_str().to_string());
    let route = format!("{} {route}", request.method());
    let signed = request.headers().get("X-Timestamp").and_then(|v| v.to_str().ok()).and_then(|v| v.parse::<f64>().ok());
    let response = next.run(request).await;
    let took = started.elapsed().as_secs_f64();
    let mut minute = MINUTE.lock().expect("request minute");
    let now = (desearch::time::now() / 60.0) as i64;
    if minute.at != now {
        if minute.at != 0 && (!minute.slow.is_empty() || minute.late.0 > 0) {
            let mut slow: Vec<_> = minute.slow.iter().collect();
            slow.sort_by_key(|(_, (count, _))| std::cmp::Reverse(*count));
            let slow: Vec<String> = slow.iter().map(|(route, (n, worst))| format!("{route} {n} (worst {worst:.1}s)")).collect();
            eprintln!(
                "requests in the last minute: {} answered; over {SLOW_INSIDE_S}s inside: {}; signed over {LATE_ARRIVAL_S}s before arriving: {} (worst {:.1}s)",
                minute.answered,
                if slow.is_empty() { "none".into() } else { slow.join(", ") },
                minute.late.0,
                minute.late.1
            );
        }
        *minute = Minute { at: now, ..Minute::default() };
    }
    minute.answered += 1;
    if took >= SLOW_INSIDE_S {
        let (count, worst) = minute.slow.entry(route).or_default();
        *count += 1;
        *worst = worst.max(took);
    }
    if let Some(lag) = signed.map(|at| arrived - at).filter(|lag| *lag >= LATE_ARRIVAL_S) {
        minute.late.0 += 1;
        minute.late.1 = minute.late.1.max(lag);
    }
    drop(minute);
    response
}

async fn method_not_allowed(request: Request, next: Next) -> Response {
    let response = next.run(request).await;
    if response.status() == StatusCode::METHOD_NOT_ALLOWED {
        return detail(StatusCode::METHOD_NOT_ALLOWED, "Method Not Allowed".into());
    }
    response
}

pub fn router(state: Arc<State>) -> Router {
    Router::new()
        .route(CLAIM_PATH, post(routes::claim))
        .route("/v1/tasks/{task_id}/complete", post(routes::complete))
        .route("/v1/tasks/{task_id}/abandon", post(routes::abandon))
        .route("/v1/validation/{task_id}/release", post(routes::release))
        .route("/v1/validation/{task_id}/score", post(routes::score))
        .route("/v1/shares", get(routes::shares))
        .route("/v1/room", get(routes::room))
        .route("/v1/admin/enqueue", post(routes::enqueue))
        .route("/v1/key", get(routes::key))
        .route("/v1/storage/probes", get(routes::storage_probes))
        .route("/v1/config", get(routes::config))
        .route("/v1/attestation", get(routes::attestation))
        .route("/v1/custody", get(routes::custody))
        .route(crate::custody::RECEIVE_PATH, post(routes::custody_receive))
        .route(crate::custody::HANDOVER_PATH, post(routes::handover))
        .route("/v1/admin/pause", post(routes::pause))
        .route("/v1/clock", get(routes::clock))
        .route("/v1/outages", get(routes::outages))
        .route("/v1/rounds", get(routes::round_list))
        .route("/v1/rounds/{round_id}", get(routes::round_view))
        .route("/v1/rounds/{round_id}/log", get(routes::round_log))
        .route("/v1/miners/{hotkey}/verdicts", get(routes::own_verdicts))
        .route("/v1/ping", get(routes::ping))
        .route("/v1/health", get(routes::health))
        .route("/v1/overview", get(logs::overview))
        .route("/v1/stats/series", get(logs::series))
        .route("/v1/stats/hours", get(logs::stat_hours))
        .route("/v1/live", get(logs::live))
        .route("/v1/miners", get(logs::miners))
        .route("/v1/miners/{hotkey}", get(logs::miner))
        .route("/v1/validators", get(logs::validators))
        .route("/v1/validators/{hotkey}", get(logs::validator))
        .route("/v1/votes", get(logs::votes))
        .route("/v1/events", get(logs::events))
        .route("/v1/tasks", get(logs::tasks))
        .route("/v1/tasks/{task_id}", get(logs::task))
        .fallback(not_found)
        .layer(middleware::from_fn_with_state(state.clone(), sign_reads))
        .layer(middleware::from_fn(method_not_allowed))
        .layer(middleware::from_fn_with_state(state.clone(), hold_writes))
        .layer(middleware::from_fn_with_state(state.clone(), limit_requests))
        .layer(cors(&state.settings.cors_origins))
        .layer(middleware::from_fn(limit_body))
        // `limit_body` sets each path's limit; axum's own 2 MB default would cut enqueues and verdicts short.
        .layer(DefaultBodyLimit::disable())
        .layer(middleware::from_fn(time_requests))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_read_only_replica_counts_as_redis_down() {
        let read_only = redis::parse_redis_value(b"-READONLY You can't write against a read only replica.\r\n").unwrap().extract_error().unwrap_err();
        assert!(redis_down(&read_only));
        let wrong = redis::parse_redis_value(b"-WRONGTYPE Operation against a key holding the wrong kind of value\r\n").unwrap().extract_error().unwrap_err();
        assert!(!redis_down(&wrong));
    }

    #[test]
    fn busy_readers_ask_the_caller_to_retry_in_a_second() {
        let ApiError::Status { status, retry_after, .. } = ApiError::from(anyhow::Error::from(Busy)) else { panic!("not a status") };
        assert_eq!((status, retry_after.as_deref()), (503, Some("1")));
    }
}
