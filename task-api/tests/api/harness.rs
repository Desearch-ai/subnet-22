//! The task API served in process against Redis db 13 and the R2 stand-in, with signed clients for each role.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Bytes;
use desearch::hotkey::{auth_headers, Hotkey};
use desearch::r2::{Bucket, Credentials};
use desearch::stub;
use redis::AsyncCommands;
use reqwest::{Method, StatusCode};
use serde_json::{json, Value};
use task_api::lifecycle;
use task_api::manifest::OPEN_LIST_KEY;
use task_api::settings::{RegistryMode, SeedsMode, Settings};
use task_api::state::{redis_from, State};
use tokio::net::TcpListener;
use tokio::sync::{Mutex, MutexGuard};

pub const REDIS_URL: &str = "redis://127.0.0.1:6379/13";
pub const MINER: &str = "//miner-api-test";
pub const RIVAL: &str = "//rival-api-test";
pub const VALIDATOR: &str = "//validator-api-test";
pub const OTHER_VALIDATOR: &str = "//validator-api-test-2";
pub const THIRD_VALIDATOR: &str = "//validator-api-test-3";
pub const ADMIN: &str = "//admin-api-test";
pub const BUCKET: &str = "subnet-22";
pub const PAGES_PREFIX: &str = "pages-bucket/";

/// A TCP relay to Redis that a test can cut, as Redis going away looks to the API.
pub struct Proxy {
    pub addr: SocketAddr,
    down: tokio::sync::watch::Sender<bool>,
}

impl Proxy {
    async fn start(target: &'static str) -> Proxy {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("a port");
        let addr = listener.local_addr().expect("an address");
        let down = tokio::sync::watch::channel(false).0;
        let watching = down.clone();
        tokio::spawn(async move {
            while let Ok((mut client, _)) = listener.accept().await {
                let mut cut = watching.subscribe();
                if *cut.borrow() {
                    continue;
                }
                tokio::spawn(async move {
                    let Ok(mut server) = tokio::net::TcpStream::connect(target).await else { return };
                    tokio::select! {
                        _ = tokio::io::copy_bidirectional(&mut client, &mut server) => {}
                        _ = cut.wait_for(|down| *down) => {}
                    }
                });
            }
        });
        Proxy { addr, down }
    }

    pub fn cut(&self) {
        self.down.send_replace(true);
    }

    pub fn restore(&self) {
        self.down.send_replace(false);
    }
}

/// Tests share one Redis database, so they run one at a time.
static SERIAL: Mutex<()> = Mutex::const_new(());

pub fn urls(count: usize) -> Vec<Value> {
    (0..count).map(|i| json!({"host": format!("site{i}.example"), "url": format!("https://site{i}.example/page/{i}")})).collect()
}

pub struct Client {
    pub hotkey: Hotkey,
    base: String,
    http: reqwest::Client,
}

/// A refused call: its status and detail.
#[derive(Debug)]
pub struct Refused {
    pub status: u16,
    pub detail: Value,
}

impl Client {
    pub fn ss58(&self) -> String {
        self.hotkey.ss58()
    }

    pub async fn call(&self, method: Method, path: &str, body: Option<Value>) -> Result<Value, Refused> {
        let bytes = body.map(|b| b.to_string().into_bytes()).unwrap_or_default();
        let nonce = format!("{:032x}", rand::random::<u128>());
        let timestamp = desearch::time::now() as i64;
        let mut request = self.http.request(method.clone(), format!("{}{path}", self.base)).header("content-type", "application/json");
        let signed_path = path.split('?').next().unwrap_or(path);
        for (name, value) in auth_headers(&self.hotkey, method.as_str(), signed_path, &bytes, timestamp, &nonce) {
            request = request.header(name, value);
        }
        let response = request.body(bytes).send().await.expect("the API answers");
        let status = response.status();
        let body: Value = response.json().await.unwrap_or(Value::Null);
        if status.is_success() {
            Ok(body)
        } else {
            Err(Refused { status: status.as_u16(), detail: body.get("detail").cloned().unwrap_or(body) })
        }
    }

    pub async fn post(&self, path: &str, body: Value) -> Result<Value, Refused> {
        self.call(Method::POST, path, Some(body)).await
    }

    pub async fn claim(&self) -> Value {
        self.call(Method::POST, "/v1/tasks/claim", None).await.expect("a claim answers")
    }

    pub async fn get(&self, path: &str) -> Result<Value, Refused> {
        self.call(Method::GET, path, None).await
    }
}

pub struct Harness {
    pub state: Arc<State>,
    pub url: String,
    pub http: reqwest::Client,
    pub redis: task_api::state::Redis,
    pub stub: stub::Stub,
    pub proxy: Proxy,
    pub miner: Client,
    pub rival: Client,
    pub validator: Client,
    pub other_validator: Client,
    pub third_validator: Client,
    pub admin: Client,
    _dir: TempDir,
    _serial: MutexGuard<'static, ()>,
}

pub struct TempDir(pub PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub fn settings(data: PathBuf, endpoint: &str) -> Settings {
    Settings {
        listen: "127.0.0.1:0".into(),
        data,
        redis_url: REDIS_URL.into(),
        claim_ttl: 180.0,
        validation_ttl: 900.0,
        poll_rate: 100.0,
        reads_per_minute: 100_000,
        log_reads_per_minute: 100_000,
        failed_writes_per_minute: 30,
        max_upload: 100_000_000,
        max_backlog: 43_200.0,
        max_attempts: 3,
        ledger_delay: 0.0,
        check_share: 1.0,
        checks_per_hour: 400.0,
        queue_target: 1200,
        task_urls: 3,
        publish_lag: 1800.0,
        publish_lag_limit: 1800.0,
        embed_tasks: false,
        embed_model: "qwen3-embedding-8b".into(),
        active_s: 3600.0,
        publish_ttl: 600.0,
        publish_tries: 5,
        r2_endpoint: endpoint.into(),
        r2_access_key: "test".into(),
        r2_secret_key: "test".into(),
        bucket: BUCKET.into(),
        prefix: String::new(),
        pages_bucket: BUCKET.into(),
        pages_prefix: PAGES_PREFIX.into(),
        registry: RegistryMode::Local { validator_uris: [VALIDATOR, OTHER_VALIDATOR, THIRD_VALIDATOR].join(",") },
        seeds: SeedsMode::Local { block_seconds: 0.02, genesis: 0.0 },
        admin_hotkeys: Vec::new(),
        admin_uris: ADMIN.into(),
        key_uri: String::new(),
        cors_origins: Vec::new(),
    }
}

impl Harness {
    pub async fn start() -> Harness {
        Harness::with(|_| {}).await
    }

    pub async fn with(tune: impl FnOnce(&mut Settings)) -> Harness {
        let serial = SERIAL.lock().await;
        let dir = TempDir(std::env::temp_dir().join(format!("task-api-test-{}", uuid::Uuid::new_v4().simple())));
        let stub = stub::start(dir.0.join("r2"), "127.0.0.1:0").await.expect("the R2 stand-in starts");
        let endpoint = format!("http://{}", stub.addr);
        let mut settings = settings(dir.0.join("data"), &endpoint);
        tune(&mut settings);
        let redis = redis_from(REDIS_URL).await.expect("Redis on localhost");
        let _: () = redis::cmd("FLUSHDB").query_async(&mut redis.clone()).await.expect("FLUSHDB");
        let proxy = Proxy::start("127.0.0.1:6379").await;
        let relayed = redis_from(&format!("redis://{}/13", proxy.addr)).await.expect("Redis through the relay");
        let (storage, pages) = buckets(&settings, &endpoint);
        let mut state = State::new(settings, relayed, storage, pages).await.expect("the state opens");
        // Running since long before any claim, so no claim counts as held through a restart.
        state.started_at = f64::NEG_INFINITY;
        let state = Arc::new(state);
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("a port");
        let addr = listener.local_addr().expect("an address");
        let router = task_api::http::router(state.clone());
        tokio::spawn(async move {
            axum::serve(listener, router.into_make_service_with_connect_info::<SocketAddr>()).await.expect("serving");
        });
        let url = format!("http://{addr}");
        let http = reqwest::Client::new();
        let client = |uri: &str| Client { hotkey: Hotkey::from_uri(uri).expect("a test key"), base: url.clone(), http: http.clone() };
        Harness {
            miner: client(MINER),
            rival: client(RIVAL),
            validator: client(VALIDATOR),
            other_validator: client(OTHER_VALIDATOR),
            third_validator: client(THIRD_VALIDATOR),
            admin: client(ADMIN),
            state,
            url,
            http,
            redis,
            stub,
            proxy,
            _dir: dir,
            _serial: serial,
        }
    }

    pub async fn public(&self, path: &str) -> (StatusCode, Value) {
        let response = self.http.get(format!("{}{path}", self.url)).send().await.expect("the API answers");
        let status = response.status();
        (status, response.json().await.unwrap_or(Value::Null))
    }

    pub async fn public_from(&self, path: &str, address: &str) -> (StatusCode, reqwest::header::HeaderMap) {
        let response = self.http.get(format!("{}{path}", self.url)).header("CF-Connecting-IP", address).send().await.expect("the API answers");
        (response.status(), response.headers().clone())
    }

    pub async fn status(&self, task_id: &str) -> String {
        self.public(&format!("/v1/tasks/{task_id}")).await.1["status"].as_str().unwrap_or_default().to_string()
    }

    pub async fn view(&self, task_id: &str) -> Value {
        self.public(&format!("/v1/tasks/{task_id}")).await.1
    }

    pub fn object(&self, key: &str) -> Option<Vec<u8>> {
        self.stub.state.object(BUCKET, key)
    }

    pub fn page(&self, key: &str) -> Option<Value> {
        self.stub.state.object(BUCKET, &format!("{PAGES_PREFIX}{key}")).and_then(|b| serde_json::from_slice(&b).ok())
    }

    pub fn json_object(&self, key: &str) -> Option<Value> {
        self.object(key).and_then(|b| serde_json::from_slice(&b).ok())
    }

    pub async fn put(&self, url: &str, body: Vec<u8>) {
        let response = self.http.put(url).header("content-type", "application/vnd.apache.parquet").body(Bytes::from(body)).send().await.expect("R2 answers");
        assert!(response.status().is_success(), "upload: {}", response.status());
    }

    pub async fn enqueue(&self, urls: Vec<Value>) -> Value {
        let enqueued = self.admin.post("/v1/admin/enqueue", json!({"urls": urls})).await.expect("enqueue");
        self.revealed().await;
        enqueued
    }

    /// Reveals rounds until none waits for its seed block; returns the queue depth.
    pub async fn revealed(&self) -> i64 {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            let pending = self.state.db.run(|conn| task_api::roundstore::unrevealed(conn, i64::MAX, 1000)).await.expect("rounds");
            if pending.is_empty() {
                return self.state.crawl.depth(&self.redis).await.expect("depth");
            }
            lifecycle::reveal_pending(&self.state).await.expect("reveal");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("no round was revealed");
    }

    /// Claims a task, uploads a file for it and completes it with the miner's counts.
    pub async fn mine(&self, miner: &Client, errors: i64) -> Value {
        let task = miner.claim().await["tasks"][0].clone();
        let body = parquet(&task);
        self.put(task["upload"]["url"].as_str().expect("an upload URL"), body.clone()).await;
        let rows = task["urls"].as_array().map_or(0, Vec::len) as i64;
        let report = json!({"key": task["upload"]["key"], "rows": rows, "ok": rows - errors, "errors": errors, "bytes": body.len()});
        miner.post(&format!("/v1/tasks/{}/complete", task["task_id"].as_str().expect("a task id")), report).await.expect("complete");
        task
    }

    /// The open list as the task API last published it.
    pub async fn open_list(&self) -> Value {
        lifecycle::settle_seeded(&self.state).await.expect("settle");
        lifecycle::publish_open(&self.state, desearch::time::now()).await.expect("publish open");
        self.json_object(OPEN_LIST_KEY).expect("an open list")
    }

    /// The oldest listed upload whose seed block has passed and `validator` has not voted on.
    pub async fn opened(&self, validator: Option<&Client>, want: Option<&str>) -> Value {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            for manifest in self.open_list().await["uploads"].as_array().cloned().unwrap_or_default() {
                let task_id = manifest["task_id"].as_str().unwrap_or_default().to_string();
                if want.is_some_and(|want| want != task_id) {
                    continue;
                }
                if let Some(validator) = validator {
                    let voted: bool = self.redis.clone().sismember(format!("vseen:{task_id}"), validator.ss58()).await.expect("voters");
                    if voted {
                        continue;
                    }
                }
                let block = manifest["seed_block"].as_i64().unwrap_or_default();
                if let Some(seed) = self.state.seeds.seed_for(block).await.expect("seed") {
                    let mut job = manifest.clone();
                    job["seed"] = seed.into();
                    return job;
                }
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        panic!("no upload opened for validation");
    }
}

fn buckets(settings: &Settings, endpoint: &str) -> (Bucket, Bucket) {
    let http = desearch::r2::client().expect("an HTTP client");
    let credentials = Credentials { access_key: "test".into(), secret_key: "test".into(), region: "auto".into() };
    let storage = Bucket::new(http.clone(), endpoint, &settings.bucket, "", credentials.clone(), Default::default()).expect("a bucket");
    let pages = Bucket::new(http, endpoint, &settings.pages_bucket, &settings.pages_prefix, credentials, Default::default()).expect("a bucket");
    (storage, pages)
}

/// A file the API takes for Parquet: it only checks the magic bytes at both ends.
pub fn parquet(task: &Value) -> Vec<u8> {
    let mut body = b"PAR1".to_vec();
    body.extend(task.to_string().as_bytes());
    body.extend(b"PAR1");
    body
}

pub fn score(verdict: &str, returned: i64, reason: &str, outcome: Option<&str>, url: &str) -> Value {
    let outcome = outcome.unwrap_or(if verdict == "pass" { "matched" } else { "mismatched" });
    let one = |name: &str| i64::from(outcome == name);
    json!({
        "returned": returned,
        "missing": 0,
        "duplicates": 0,
        "sampled": 1,
        "matched": one("matched"),
        "mismatched": one("mismatched"),
        "unverifiable": one("unverifiable"),
        "errors_confirmed": one("errors_confirmed"),
        "errors_unconfirmed": 0,
        "reextract_mismatch": 0,
        "error_rows": if outcome == "errors_confirmed" { returned } else { 0 },
        "verdict": verdict,
        "reason": reason,
        "samples": [{
            "url": url,
            "outcome": outcome,
            "similarity": if outcome == "matched" { 0.93 } else { 0.12 },
            "miner_status": 200,
            "validator_status": 200,
            "miner_chars": 40,
            "validator_chars": 41,
        }],
    })
}

/// Votes on the task, or on the oldest upload this validator has not voted on, sampling the task's own URLs.
pub async fn judged(h: &Harness, validator: &Client, verdict: &str, task_id: Option<&str>, overrides: Value) -> (Value, Result<Value, Refused>) {
    let job = h.opened(Some(validator), task_id).await;
    let urls: Vec<Value> = job["urls"].as_array().cloned().unwrap_or_default();
    let mut body = score(verdict, urls.len() as i64, if verdict == "pass" { "ok" } else { "content_mismatch" }, None, "");
    for (name, value) in overrides.as_object().cloned().unwrap_or_default() {
        body[name] = value;
    }
    if let Some(samples) = body["samples"].as_array_mut() {
        for (sample, url) in samples.iter_mut().zip(urls) {
            sample["url"] = url;
        }
    }
    let path = format!("/v1/validation/{}/score", job["task_id"].as_str().unwrap_or_default());
    let answer = validator.post(&path, body).await;
    (job, answer)
}

pub fn task_id(task: &Value) -> String {
    task["task_id"].as_str().expect("a task id").to_string()
}

pub fn first_url(task: &Value) -> String {
    task["urls"][0].as_str().expect("a URL").to_string()
}
