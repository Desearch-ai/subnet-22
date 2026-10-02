//! The task API calls the dispatcher makes, signed as `desearch.client.TaskApiClient` signs them.

use std::time::Duration;

use anyhow::Result;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::hotkey::{auth_headers, Hotkey};

const TIMEOUT: Duration = Duration::from_secs(60);

/// How many tasks the queue can take now.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
pub struct Room {
    pub room_tasks: i64,
    #[serde(default)]
    pub queue: i64,
    #[serde(default)]
    pub unrevealed: i64,
    #[serde(default)]
    pub refusing: bool,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct QueuedUrl {
    pub host: String,
    pub url: String,
}

#[derive(Serialize)]
struct Enqueue<'a> {
    urls: &'a [QueuedUrl],
    batch_id: &'a str,
}

#[derive(Debug)]
pub enum ApiError {
    /// No answer: the batch may or may not have been taken.
    Unreachable(String),
    Refused(u16, String),
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApiError::Unreachable(error) => write!(f, "{error}"),
            ApiError::Refused(status, detail) => write!(f, "HTTP {status}: {detail}"),
        }
    }
}

impl std::error::Error for ApiError {}

pub struct TaskApi {
    base: String,
    http: reqwest::Client,
    hotkey: Hotkey,
}

impl TaskApi {
    pub fn new(base: &str, hotkey: Hotkey) -> Result<Self> {
        let http = reqwest::Client::builder().timeout(TIMEOUT).pool_max_idle_per_host(0).build()?;
        Ok(TaskApi { base: base.trim_end_matches('/').to_string(), http, hotkey })
    }

    pub fn hotkey(&self) -> String {
        self.hotkey.ss58()
    }

    pub async fn room(&self) -> Result<Room, ApiError> {
        let raw = self.send(reqwest::Method::GET, "/v1/room", Vec::new()).await?;
        serde_json::from_slice(&raw).map_err(|error| ApiError::Unreachable(format!("unreadable room: {error}")))
    }

    /// Queue one batch; the API takes a batch_id it has seen as already done.
    pub async fn enqueue(&self, urls: &[QueuedUrl], batch_id: &str) -> Result<Value, ApiError> {
        let body = serde_json::to_vec(&Enqueue { urls, batch_id }).expect("plain JSON");
        let raw = self.send(reqwest::Method::POST, "/v1/admin/enqueue", body).await?;
        Ok(serde_json::from_slice(&raw).unwrap_or(Value::Null))
    }

    async fn send(&self, method: reqwest::Method, path: &str, body: Vec<u8>) -> Result<Vec<u8>, ApiError> {
        let url = reqwest::Url::parse(&format!("{}{path}", self.base)).map_err(|error| ApiError::Unreachable(error.to_string()))?;
        let mut nonce = [0u8; 16];
        rand::thread_rng().fill_bytes(&mut nonce);
        let nonce: String = nonce.iter().map(|b| format!("{b:02x}")).collect();
        let now = chrono::Utc::now().timestamp();
        let mut request = self.http.request(method.clone(), url.clone());
        for (name, value) in auth_headers(&self.hotkey, method.as_str(), url.path(), &body, now, &nonce) {
            request = request.header(name, value);
        }
        if !body.is_empty() {
            request = request.header("Content-Type", "application/json").body(body);
        }
        let response = request.send().await.map_err(|error| ApiError::Unreachable(causes(&error)))?;
        let status = response.status();
        let raw = response.bytes().await.map_err(|error| ApiError::Unreachable(causes(&error)))?;
        if status.is_success() {
            return Ok(raw.to_vec());
        }
        let detail = serde_json::from_slice::<Value>(&raw)
            .ok()
            .and_then(|v| v.get("detail").map(|d| d.as_str().map_or_else(|| d.to_string(), str::to_string)))
            .unwrap_or_else(|| String::from_utf8_lossy(&raw).into_owned());
        Err(ApiError::Refused(status.as_u16(), detail))
    }
}

/// An error with every cause behind it, as reqwest keeps the useful part in its sources.
fn causes(error: &dyn std::error::Error) -> String {
    let mut text = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        text.push_str(": ");
        text.push_str(&cause.to_string());
        source = cause.source();
    }
    text
}
