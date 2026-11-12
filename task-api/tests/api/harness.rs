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

/// The Redis the tests use, `127.0.0.1:6379` unless `TASK_API_TEST_REDIS` (or `TASK_API_TEST_REDIS_ADDR`) names another, so
/// checkouts testing at once can each have their own.
pub fn redis_addr() -> &'static str {
    static ADDR: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
        std::env::var("TASK_API_TEST_REDIS").or_else(|_| std::env::var("TASK_API_TEST_REDIS_ADDR")).unwrap_or_else(|_| "127.0.0.1:6379".into())
    });
    &ADDR
}

pub fn redis_url(db: u8) -> String {
    format!("redis://{}/{db}", redis_addr())
}

/// Redis db 13 at `redis_addr`, which A's tests name as one.
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
        self.call_at(method, path, body, desearch::time::now() as i64).await
    }

    /// A request signed as sent at `timestamp`, as a client whose clock differs from this machine's signs it.
    pub async fn call_at(&self, method: Method, path: &str, body: Option<Value>, timestamp: i64) -> Result<Value, Refused> {
        let bytes = body.map(|b| b.to_string().into_bytes()).unwrap_or_default();
        let nonce = format!("{:032x}", rand::random::<u128>());
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

    /// A signed POST and the hash naming it, as a relaying validator's witness gives it.
    pub async fn post_hashed(&self, path: &str, body: Option<Value>) -> (Result<Value, Refused>, String) {
        let bytes = body.map(|b| b.to_string().into_bytes()).unwrap_or_default();
        let nonce = format!("{:032x}", rand::random::<u128>());
        let timestamp = desearch::time::now() as i64;
        let headers = auth_headers(&self.hotkey, "POST", path, &bytes, timestamp, &nonce);
        let hashed = task_api::auth::request_hash("POST", path, &bytes, &self.ss58(), &timestamp.to_string(), &nonce, &headers[3].1);
        let mut request = self.http.post(format!("{}{path}", self.base)).header("content-type", "application/json");
        for (name, value) in headers {
            request = request.header(name, value);
        }
        let response = request.body(bytes).send().await.expect("the API answers");
        let status = response.status();
        let body: Value = response.json().await.unwrap_or(Value::Null);
        let answer = if status.is_success() { Ok(body) } else { Err(Refused { status: status.as_u16(), detail: body.get("detail").cloned().unwrap_or(body) }) };
        (answer, hashed)
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
        redis_url: redis_url(13),
        claim_ttl: 180.0,
        embed_claim_ttl: 300.0,
        validation_ttl: 900.0,
        poll_rate: 100.0,
        reads_per_minute: 100_000,
        log_reads_per_minute: 100_000,
        failed_writes_per_minute: 30,
        max_upload: 100_000_000,
        max_backlog: 43_200.0,
        max_attempts: 3,
        max_handbacks: 5,
        ledger_delay: 0.0,
        check_share: 1.0,
        checks_per_hour: 400.0,
        embed_check_share: 1.0,
        embed_checks_per_hour: 600.0,
        queue_target: 1200,
        embed_queue_target: 100,
        embed_input_max_age: 72_000.0,
        release_spread: 0.0,
        task_urls: 3,
        embed_tasks: false,
        embed_model: "desearch-4b-v1".into(),
        active_s: 3600.0,
        publish_ttl: 600.0,
        publish_tries: 5,
        r2_endpoint: endpoint.into(),
        r2_access_key: "test".into(),
        r2_secret_key: "test".into(),
        r2_region: "auto".into(),
        public_objects: false,
        bucket: BUCKET.into(),
        prefix: String::new(),
        pages_bucket: BUCKET.into(),
        pages_prefix: PAGES_PREFIX.into(),
        pages_store: None,
        registry: RegistryMode::Local { validator_uris: [VALIDATOR, OTHER_VALIDATOR, THIRD_VALIDATOR].join(",") },
        validator_min_stake: 10_000.0,
        validator_min_alpha: 20.0,
        seeds: SeedsMode::Local { block_seconds: 0.02, genesis: 0.0 },
        admin_hotkeys: Vec::new(),
        admin_uris: ADMIN.into(),
        journal_every: 30.0,
        journal_shares_every: 600.0,
        snapshot_every: 21_600.0,
        journal_keep: 8.0 * 86_400.0,
        anchor: None,
        key_uri: String::new(),
        cors_origins: Vec::new(),
        canary_every: 0.0,
        upload_lock_s: 0.0,
        storage_witness_quorum: 2,
        storage_witness_window: 300.0,
        client_address_header: task_api::auth::CLIENT_ADDRESS_HEADER.into(),
        chain: Default::default(),
        tee: Default::default(),
        storage_probe_s: 30.0,
        clock_skew_s: 30.0,
        finality_stall_blocks: 10,
        objective_quorum: 1,
        embed_references: 2,
        loadability_samples: 30,
        neutral_uploads: 3,
        neutral_share: 0.5,
        incentive_full: 0.5,
        incentive_grace_h: 24,
        crawl_emission: 0.2,
        embed_emission: 0.5,
        wrong_reports: 5,
        wrong_reports_window_s: 86_400.0,
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
        let redis = redis_from(&redis_url(13)).await.expect("Redis on localhost");
        let _: () = redis::cmd("FLUSHDB").query_async(&mut redis.clone()).await.expect("FLUSHDB");
        let proxy = Proxy::start(redis_addr()).await;
        let relayed = redis_from(&format!("redis://{}/13", proxy.addr)).await.expect("Redis through the relay");
        let (storage, pages) = buckets(&settings, &endpoint);
        let state = State::new(settings, relayed, storage, pages).await.expect("the state opens");
        // Running since long before any claim, so no claim counts as held through a restart.
        state.set_started_at(f64::NEG_INFINITY);
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

    /// Serves another API on the same buckets but its own Redis db and data directory, as a second CVM would; returns it and its URL.
    pub async fn sibling(&self, redis_db: u8, tune: impl FnOnce(&mut Settings)) -> (Arc<State>, String, TempDir) {
        let dir = TempDir(std::env::temp_dir().join(format!("task-api-sibling-{}", uuid::Uuid::new_v4().simple())));
        let endpoint = format!("http://{}", self.stub.addr);
        let mut settings = settings(dir.0.join("data"), &endpoint);
        settings.redis_url = redis_url(redis_db);
        tune(&mut settings);
        let redis = redis_from(&settings.redis_url).await.expect("Redis on localhost");
        let _: () = redis::cmd("FLUSHDB").query_async(&mut redis.clone()).await.expect("FLUSHDB");
        let (storage, pages) = buckets(&settings, &endpoint);
        let state = State::new(settings, redis, storage, pages).await.expect("the sibling opens");
        state.set_started_at(f64::NEG_INFINITY);
        let state = Arc::new(state);
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("a port");
        let url = format!("http://{}", listener.local_addr().expect("an address"));
        let router = task_api::http::router(state.clone());
        tokio::spawn(async move {
            axum::serve(listener, router.into_make_service_with_connect_info::<SocketAddr>()).await.expect("serving");
        });
        (state, url, dir)
    }

    /// A client for another API, signing as one of this harness's roles.
    pub fn client_of(&self, base: &str, uri: &str) -> Client {
        Client { hotkey: Hotkey::from_uri(uri).expect("a test key"), base: base.to_string(), http: self.http.clone() }
    }

    /// Serves another copy started now on the same Redis and database, as after a restart; returns its URL.
    pub async fn restarted(&self) -> String {
        let state =
            State::new(self.state.settings.clone(), self.redis.clone(), self.state.storage.clone(), self.state.pages.clone()).await.expect("the copy opens");
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("a port");
        let url = format!("http://{}", listener.local_addr().expect("an address"));
        let router = task_api::http::router(Arc::new(state));
        tokio::spawn(async move {
            axum::serve(listener, router.into_make_service_with_connect_info::<SocketAddr>()).await.expect("serving");
        });
        url
    }

    pub fn client(&self, uri: &str, base: &str) -> Client {
        Client { hotkey: Hotkey::from_uri(uri).expect("a test key"), base: base.into(), http: self.http.clone() }
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

    /// Closes the rounds whose tasks are all done, then every other round some finalized crawl upload waits on, through the
    /// API's own close, as if the round's last task were done: their accounts settle on the round as it stands. Returns the
    /// rounds closed early.
    pub async fn settle(&self) -> Vec<String> {
        lifecycle::close_finished(&self.state).await.expect("close finished rounds");
        let held: Vec<String> = self
            .state
            .db
            .run(|conn| {
                let mut statement = conn.prepare("SELECT DISTINCT round_id FROM round_uploads ORDER BY round_id")?;
                let rounds = statement.query_map([], |row| row.get(0))?.collect::<rusqlite::Result<_>>()?;
                Ok(rounds)
            })
            .await
            .expect("held rounds");
        for round_id in &held {
            let serving = self.state.current.lock().expect("current rounds").clone();
            lifecycle::close_round(&self.state, round_id).await.expect("close the round");
            // Closed before its last task: the round opens again and serves the tasks it has left as the test goes on; its
            // real close settles whatever finalizes later.
            let open: i64 = self.redis.clone().scard(lifecycle::open_round_key(round_id)).await.expect("open tasks");
            if open > 0 {
                let reopened = round_id.clone();
                self.state
                    .db
                    .run(move |conn| {
                        conn.execute("UPDATE rounds SET closed_at = NULL WHERE round_id = ?", [&reopened])?;
                        conn.execute("DELETE FROM anchors WHERE round_id = ?", [&reopened])?;
                        Ok(())
                    })
                    .await
                    .expect("reopen the round");
                *self.state.current.lock().expect("current rounds") = serving;
            }
        }
        held
    }

    /// Finalizes every open upload as if its validation deadline had passed, on the votes it has, and settles the rounds its
    /// crawl uploads wait on: a fail on validators' reports is decided only then. Returns the uploads finalized.
    pub async fn past_deadline(&self) -> Vec<String> {
        // A deadline seconds away is waited for; one far off is moved to now in Redis, which the log does not see.
        let mut latest = 0.0_f64;
        for task in self.state.validation.open_ids(&self.redis, 1000).await.expect("open uploads") {
            let Some(mut job) = self.state.validation.job(&self.redis, &task).await.expect("an open job") else { continue };
            let deadline = job["deadline"].as_f64().unwrap_or(0.0);
            if deadline - desearch::time::now() > 10.0 {
                job["deadline"] = (self.state.rule_now().await.expect("rule time") - 1.0).into();
                let _: () = self.redis.clone().set(format!("vjob:{task}"), job.to_string()).await.expect("move the deadline");
            } else {
                latest = latest.max(deadline);
            }
        }
        while self.state.rule_now().await.expect("rule time") < latest {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        let at = self.state.rule_now().await.expect("rule time");
        let finalized = lifecycle::finalize_due(&self.state, at).await.expect("finalize at the deadline");
        self.settle().await;
        finalized
    }

    /// A finalized crawl upload's answer once its round closed: its accounts as they settled.
    pub async fn settled_answer(&self, mut answer: Value) -> Value {
        if answer["verdict"] != lifecycle::HELD {
            return answer;
        }
        self.settle().await;
        let task_id = answer["task_id"].as_str().unwrap_or_default().to_string();
        let found = self
            .state
            .db
            .run(move |conn| {
                Ok(conn.query_row(
                    "SELECT v.verdict, v.reason, v.credited, v.miner FROM validations v WHERE v.task_id = ? ORDER BY v.id DESC LIMIT 1",
                    [&task_id],
                    |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, i64>(2)?, row.get::<_, String>(3)?)),
                )?)
            })
            .await
            .expect("the settled verdict");
        let (verdict, reason, credited, miner) = found;
        let budget = self.state.db.run(move |conn| task_api::budgets::get(conn, &miner, task_api::budgets::CRAWL)).await.expect("budget").budget;
        answer["verdict"] = verdict.into();
        answer["reason"] = reason.into();
        answer["credited"] = credited.into();
        answer["miner_budget"] = budget.into();
        answer
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

/// An embed upload of the size the API expects: one float16 vector of the model per text.
pub fn vectors(texts: u64) -> Vec<u8> {
    vec![0; task_api::embedding::upload_bytes(task_api::embedding::DESEARCH_4B_V1, texts) as usize]
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

/// Overrides that fail a check on a fact of the file, which charges the miner, not on the live pages, which does not.
pub fn objective_fail() -> Value {
    json!({"reason": "text_not_from_html", "reextract_mismatch": 1, "integrity_checked": 1})
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
    // A crawl verdict's accounts wait for its round to close; the round is closed at once so a test reads them.
    let answer = match answer {
        Ok(answer) => Ok(h.settled_answer(answer).await),
        refused => refused,
    };
    (job, answer)
}

/// A key the claim drew at random: `uploads/`, 32 hex characters and the kind's suffix, naming no task or miner.
pub fn is_random_upload_key(key: &str, suffix: &str) -> bool {
    key.strip_prefix("uploads/").and_then(|name| name.strip_suffix(suffix)).is_some_and(|name| name.len() == 32 && name.bytes().all(|b| b.is_ascii_hexdigit()))
}

pub fn task_id(task: &Value) -> String {
    task["task_id"].as_str().expect("a task id").to_string()
}

pub fn first_url(task: &Value) -> String {
    task["urls"][0].as_str().expect("a URL").to_string()
}
