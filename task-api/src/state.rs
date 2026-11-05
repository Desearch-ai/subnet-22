//! What every request and the janitor share: settings, Redis, SQLite, the buckets, the chain and the receipt key.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result};
use desearch::canonical::hex;
use desearch::hotkey::Hotkey;
use desearch::r2::{Bucket, Credentials};
use desearch::time::now;
use redis::AsyncCommands;
use serde_json::{json, Value};

use crate::attestation::{self, Tee};
use crate::budgets::{self, CRAWL, EMBED};
use crate::chain::Chain;
use crate::clock::{self, Clock, Tick};
use crate::custody::Custody;
use crate::db::{Db, Reader};
use crate::flow::PublishRate;
use crate::logs::Cache;
use crate::py::{canonical, round_to};
use crate::queues::{PublishQueue, TaskQueue, ValidationQueue};
use crate::registry::{self, addresses, Registry};
use crate::roundlog::{self, Receipt};
use crate::rounds::UPLOAD_GRACE_S;
use crate::rules;
use crate::seeds::Seeds;
use crate::settings::{KeySource, RegistryMode, SeedsMode, Settings};
use crate::{db, journal, roundstore};

pub type Redis = redis::aio::ConnectionManager;

/// How often the incentive cap reads every hotkey's share of the work again.
const WORK_REFRESH: Duration = Duration::from_secs(60);

pub const POLL_WINDOW_S: f64 = 10.0;
pub const MINUTE: f64 = 60.0;
const DEV_RECEIPT_KEY: &str = "//TaskApi";

pub struct State {
    pub settings: Settings,
    pub redis: Redis,
    pub db: Db,
    pub reader: Reader,
    /// Every other read, apart from the log pages so a busy dashboard cannot hold up a claim.
    pub lookups: Reader,
    pub cache: Cache,
    pub storage: Bucket,
    pub pages: Bucket,
    pub registry: Arc<Registry>,
    pub chain: Option<Arc<Chain>>,
    pub admins: HashSet<String>,
    pub seeds: Seeds,
    pub key: Arc<Hotkey>,
    /// The CVM this copy runs in, when its key was made there.
    pub tee: Option<Arc<Tee>>,
    /// Whether this copy holds the log, and the pause.
    pub custody: Custody,
    pub crawl: TaskQueue,
    pub embed: TaskQueue,
    pub validation: ValidationQueue,
    pub publish: PublishQueue,
    pub publish_rate: Mutex<PublishRate>,
    /// The newest revealed round of each kind, where refusals are logged.
    pub current: Mutex<HashMap<String, String>>,
    /// Each open upload's manifest and deadline, read once instead of on every listing.
    pub open_manifests: Mutex<HashMap<String, (Value, f64)>>,
    pub open_listed: Mutex<Option<Vec<String>>>,
    /// Claims that expired before this rule time plus one claim's time were held through a restart: the wall time at start,
    /// lowered to the rule time of the first finalized block this process reads.
    started_at: AtomicU64,
    /// How a finalized height becomes rule time; fixed once read.
    /// How heights become rule time, read once; a copy handed or appointed to the log forgets its own and reads the log's.
    clock: std::sync::Mutex<Option<Clock>>,
    clock_read: tokio::sync::Mutex<()>,
    /// Every hotkey's share of each pool and its first hour of work, read at most once a minute for the incentive cap.
    pub work: Mutex<Option<(std::time::Instant, Arc<budgets::Work>)>>,
    /// Set when storage failed a request, so the janitor probes it without waiting for its turn.
    pub probe_storage: AtomicBool,
    /// Milliseconds the host clock runs ahead of the chain's once it is off by more than `TASK_API_CLOCK_SKEW_S`, else 0: signed
    /// requests are timed by the chain then.
    clock_offset_ms: AtomicI64,
    /// The last clock check that agreed with the chain: a clock outage is dated from it.
    pub clock_ok: Mutex<Option<Tick>>,
    /// Set on shutdown so /v1/ping fails and the proxy stops routing here before the server stops.
    pub draining: AtomicBool,
    /// Completions running on after their caller may have stopped waiting.
    pub detached: AtomicUsize,
}

/// What a `start` entry holds: when this copy started, its version and key, and every setting it runs with (secrets only by name).
pub fn start_input(settings: &Settings, signer: &str, started_at: f64) -> Value {
    let public = settings.public();
    json!({
        "started_at": started_at,
        "version": env!("CARGO_PKG_VERSION"),
        "commit": public["commit"],
        "signer": signer,
        "settings": settings.journaled(),
        "secrets_set": public["secrets_set"],
    })
}

/// Signs receipts and writes them to the round log in the caller's transaction; returns each as the caller is shown it.
pub fn record_receipts(conn: &rusqlite::Connection, key: &Hotkey, receipts: Vec<Receipt>, served_at: f64) -> Result<Vec<Value>> {
    let mut shown = Vec::with_capacity(receipts.len());
    for mut receipt in receipts {
        // Another copy of the API may have sealed the round this one still names.
        if receipt.outcome == "refused" && roundlog::anchored_root(conn, &receipt.round_id)?.is_some() {
            receipt.round_id.clear();
        }
        let body = receipt.body();
        let signature = hex(&key.sign(&canonical(&body)));
        roundlog::record(conn, &receipt, &signature, served_at)?;
        shown.push(json!({"body": body, "signature": signature}));
    }
    Ok(shown)
}

pub async fn redis_from(url: &str) -> Result<Redis> {
    let client = redis::Client::open(url).with_context(|| format!("TASK_API_REDIS {url:?}"))?;
    let config = redis::aio::ConnectionManagerConfig::new()
        .set_connection_timeout(Some(Duration::from_secs(2)))
        .set_response_timeout(Some(Duration::from_secs(10)))
        .set_number_of_retries(1);
    Ok(redis::aio::ConnectionManager::new_with_config(client, config).await?)
}

/// Refused completions and abandons of one hotkey logged per minute; more are answered 429 and not logged.
pub const REFUSALS_LOGGED_PER_MINUTE: i64 = 30;

/// Log pages queued at once before more are turned away.
const LOG_READS_WAITING: usize = 32;
const LOG_THREADS: usize = 2;
const LOOKUP_THREADS: usize = 4;
const LOOKUPS_WAITING: usize = 1024;

/// The last best block any copy read, and when; a restart counts on from it instead of waiting on the chain.
pub const BEST_BLOCK_KEY: &str = "chain:best";
/// Older than this, a kept block may have drifted from the chain's, so it is read again.
const KEPT_BLOCK_S: f64 = 600.0;

async fn kept_block(redis: &Redis) -> Option<(i64, f64)> {
    let raw: Option<String> = redis.clone().get(BEST_BLOCK_KEY).await.ok()?;
    let kept: Value = serde_json::from_str(&raw?).ok()?;
    Some((kept["block"].as_i64()?, kept["at"].as_f64()?))
}

impl State {
    pub async fn keep_best_block(&self) -> Result<()> {
        if let Some((block, at)) = self.chain.as_ref().and_then(|chain| chain.last_read()) {
            let _: () = self.redis.clone().set(BEST_BLOCK_KEY, json!({"block": block, "at": at}).to_string()).await?;
        }
        Ok(())
    }

    pub async fn new(settings: Settings, redis: Redis, storage: Bucket, pages: Bucket) -> Result<State> {
        std::fs::create_dir_all(&settings.data).with_context(|| format!("TASK_API_DATA {:?}", settings.data))?;
        let path = settings.data.join(db::FILE);
        let db = Db::open(&path)?;
        let reader = Reader::open(&path, "logs", LOG_THREADS, LOG_READS_WAITING)?;
        let lookups = Reader::open(&path, "lookups", LOOKUP_THREADS, LOOKUPS_WAITING)?;
        let current = db.run(roundstore::latest_revealed).await?;
        let network = match (&settings.registry, &settings.seeds) {
            (RegistryMode::Chain { network, .. }, _) | (_, SeedsMode::Chain { network }) => Some(network),
            _ => None,
        };
        let chain = network.map(|network| Chain::with(network, &settings.chain)).transpose()?.map(Arc::new);
        let registry = match &settings.registry {
            RegistryMode::Chain { .. } => Registry::chain(),
            RegistryMode::Local { validator_uris } => Registry::local(addresses(validator_uris)?),
        };
        let kept = db.run(registry::saved).await?;
        if !kept.is_empty() {
            registry.restore(kept);
        }
        if let (Some(chain), Some((block, at))) = (&chain, kept_block(&redis).await) {
            if now() - at < KEPT_BLOCK_S {
                chain.assume(block, at);
            }
        }
        let seeds = match &settings.seeds {
            SeedsMode::Chain { .. } => Seeds::Chain(chain.clone().context("the seeds need the chain")?),
            SeedsMode::Local { block_seconds, genesis } => Seeds::Local { block_seconds: *block_seconds, genesis: *genesis },
        };
        let mut admins: HashSet<String> = settings.admin_hotkeys.iter().cloned().collect();
        admins.extend(addresses(&settings.admin_uris)?);
        let (key, tee) = match settings.tee.key_source {
            KeySource::Uri => {
                (Hotkey::from_uri(if settings.key_uri.is_empty() { DEV_RECEIPT_KEY } else { &settings.key_uri }).context("TASK_API_KEY_URI")?, None)
            }
            KeySource::Cvm => (
                attestation::cvm_key(&settings.data).context("the CVM's key")?,
                Some(Arc::new(Tee::open(&settings.data, &settings.tee.dstack_socket, &settings.tee.tls_cert)?)),
            ),
        };
        let started_at = now();
        let started = start_input(&settings, &key.ss58(), started_at);
        db.run(move |conn| {
            let (last, _) = journal::head(conn)?;
            journal::append(conn, journal::Record::new("start", started_at, started, Value::Null))?;
            // The first log starts from the accounts as they are, so a replay can follow on from them.
            if last == 0 {
                crate::snapshot::record(conn, started_at)?;
            }
            Ok(())
        })
        .await?;
        let custody = Custody::open(&settings.data, settings.tee.custody)?;
        custody.note_pause(db.run(crate::custody::current_pause).await?);
        Ok(State {
            redis,
            db,
            reader,
            lookups,
            cache: Cache::default(),
            storage,
            pages,
            registry: Arc::new(registry),
            chain,
            admins,
            seeds,
            key: Arc::new(key),
            tee,
            custody,
            crawl: TaskQueue::new("crawl", settings.claim_ttl_for(CRAWL) + UPLOAD_GRACE_S),
            embed: TaskQueue::new("embed", settings.claim_ttl_for(EMBED) + UPLOAD_GRACE_S),
            validation: ValidationQueue { active_s: settings.active_s },
            publish: PublishQueue { claim_ttl: settings.publish_ttl, max_tries: settings.publish_tries },
            publish_rate: Mutex::new(PublishRate::default()),
            current: Mutex::new(current),
            open_manifests: Mutex::default(),
            open_listed: Mutex::default(),
            started_at: AtomicU64::new(started_at.to_bits()),
            clock: std::sync::Mutex::default(),
            clock_read: tokio::sync::Mutex::default(),
            work: Mutex::default(),
            probe_storage: AtomicBool::new(false),
            clock_offset_ms: AtomicI64::new(0),
            clock_ok: Mutex::default(),
            draining: AtomicBool::new(false),
            detached: AtomicUsize::new(0),
            settings,
        })
    }

    /// The buckets the settings name, both on the R2 endpoint unless the pages bucket has its own.
    pub fn buckets(settings: &Settings) -> Result<(Bucket, Bucket)> {
        let http = desearch::r2::client()?;
        let credentials =
            Credentials { access_key: settings.r2_access_key.clone(), secret_key: settings.r2_secret_key.clone(), region: settings.r2_region.clone() };
        let storage = Bucket::new(http.clone(), &settings.r2_endpoint, &settings.bucket, &settings.prefix, credentials.clone(), Default::default())?
            .public_objects(settings.public_objects);
        let (endpoint, credentials) = match &settings.pages_store {
            Some(own) => (&own.endpoint, Credentials { access_key: own.access_key.clone(), secret_key: own.secret_key.clone(), region: own.region.clone() }),
            None => (&settings.r2_endpoint, credentials),
        };
        let pages = Bucket::new(http, endpoint, &settings.pages_bucket, &settings.pages_prefix, credentials, Default::default())?;
        Ok((storage, pages))
    }

    /// How heights become rule time: from the genesis on a local chain, from the saved anchor on a real one.
    pub async fn clock(&self) -> Result<Clock> {
        if let Some(clock) = *self.clock.lock().expect("clock") {
            return Ok(clock);
        }
        let _one = self.clock_read.lock().await;
        if let Some(clock) = *self.clock.lock().expect("clock") {
            return Ok(clock);
        }
        let block_seconds = self.seeds.block_seconds();
        let clock = match &self.settings.seeds {
            SeedsMode::Local { genesis, .. } => Clock { anchor_block: 0, anchor_at: *genesis, block_seconds },
            SeedsMode::Chain { .. } => {
                let (block, at) = (self.seeds.height().await?, now());
                let (anchor_block, anchor_at) = self.db.run(move |conn| clock::anchor(conn, block, at)).await?;
                Clock { anchor_block, anchor_at, block_seconds }
            }
        };
        *self.clock.lock().expect("clock") = Some(clock);
        Ok(clock)
    }

    /// Reads the clock's anchor from SQLite again: after a handover or an appointment it is the log's, not the one this copy saw first.
    pub fn forget_clock(&self) {
        *self.clock.lock().expect("clock") = None;
    }

    /// The finalized height now and its rule time, which every deadline, window and lockout is measured on.
    /// Seconds the host clock runs ahead of the chain's while they disagree, else 0.
    pub fn clock_offset(&self) -> f64 {
        self.clock_offset_ms.load(Ordering::Acquire) as f64 / 1000.0
    }

    pub fn set_clock_offset(&self, seconds: f64) {
        self.clock_offset_ms.store((seconds * 1000.0).round() as i64, Ordering::Release);
    }

    /// The time signed requests are judged by: the host's, or the chain's while the host clock is off it.
    pub fn request_now(&self) -> f64 {
        now() - self.clock_offset()
    }

    pub async fn tick(&self) -> Result<Tick> {
        let block = self.seeds.height().await?;
        let tick = self.clock().await?.tick(block);
        let lowered = self.started_at.fetch_update(Ordering::AcqRel, Ordering::Acquire, |bits| (tick.at < f64::from_bits(bits)).then(|| tick.at.to_bits()));
        if let Ok(was) = lowered {
            // A lapse forgiven for this restart names the start it was decided on, so the log holds the one it was lowered to.
            let input = json!({"started_at": tick.at, "block": tick.block, "was": f64::from_bits(was)});
            self.db
                .run(move |conn| journal::append(conn, journal::Record::new("started", input["started_at"].as_f64().unwrap_or_default(), input, Value::Null)))
                .await?;
        }
        Ok(tick)
    }

    pub async fn rule_now(&self) -> Result<f64> {
        Ok(self.tick().await?.at)
    }

    /// The rule time this process started at.
    pub fn started_at(&self) -> f64 {
        f64::from_bits(self.started_at.load(Ordering::Acquire))
    }

    /// Moves the start, as a restart would; tests use it.
    pub fn set_started_at(&self, at: f64) {
        self.started_at.store(at.to_bits(), Ordering::Release);
    }

    /// The budget a miner's on-chain incentive allows, `CEILING` where there is nothing to compare: no metagraph incentive,
    /// no counted work, or work younger than the grace.
    pub async fn budget_cap(&self, hotkey: &str, at: f64) -> Result<i64> {
        let params = self.settings.rules();
        let Some(incentive) = self.registry.registered(hotkey).and_then(|entry| entry.incentive).filter(|_| params.incentive_full > 0.0) else {
            return Ok(budgets::CEILING);
        };
        let fresh = self.work.lock().expect("work").as_ref().filter(|(read, _)| read.elapsed() < WORK_REFRESH).map(|(_, work)| work.clone());
        let work = match fresh {
            Some(work) => work,
            None => {
                let work = Arc::new(self.lookups.read(move |conn| budgets::work(conn, budgets::SHARE_WINDOW_H, at)).await?);
                *self.work.lock().expect("work") = Some((std::time::Instant::now(), work.clone()));
                work
            }
        };
        let Some((shares, first)) = work.get(hotkey) else { return Ok(budgets::CEILING) };
        let expected = rules::expected_incentive(shares, &params);
        if *first > budgets::hour_of(at) - params.incentive_grace_h || expected <= 0.0 {
            return Ok(budgets::CEILING);
        }
        Ok(rules::incentive_cap(Some(incentive / expected), &params))
    }

    pub fn tasks(&self, kind: &str) -> &TaskQueue {
        if kind == "embed" {
            &self.embed
        } else {
            &self.crawl
        }
    }

    pub fn current_round(&self, kind: &str) -> String {
        self.current.lock().expect("current rounds").get(kind).cloned().unwrap_or_default()
    }

    pub fn signer(&self) -> String {
        self.key.ss58()
    }

    /// The development key a copy signs with when given no key at all.
    pub fn dev_signer() -> String {
        Hotkey::from_uri(DEV_RECEIPT_KEY).map(|key| key.ss58()).unwrap_or_default()
    }

    pub fn sign(&self, message: &[u8]) -> String {
        hex(&self.key.sign(message))
    }

    pub async fn payload(&self, task_id: &str) -> Result<Option<Value>> {
        let found: Option<String> = self.redis.clone().get(format!("task:{task_id}")).await?;
        Ok(found.and_then(|raw| serde_json::from_str(&raw).ok()))
    }

    pub async fn claim_holder(&self, task_id: &str) -> Result<Option<String>> {
        Ok(self.redis.clone().get(format!("claim:{task_id}")).await?)
    }

    pub async fn next_seq(&self) -> Result<i64> {
        Ok(self.redis.clone().incr("log:seq", 1).await?)
    }

    /// Signs receipts and writes them to the log together; returns each as the caller is shown it.
    pub async fn record(&self, receipts: Vec<Receipt>) -> Result<Vec<Value>> {
        let key = self.key.clone();
        self.db
            .run(move |conn| {
                let shown = record_receipts(conn, &key, receipts, now())?;
                journal::append(conn, journal::Record::new("receipt", now(), Value::Null, json!({"receipts": shown})))?;
                Ok(shown)
            })
            .await
    }

    pub async fn record_one(&self, receipt: Receipt) -> Result<Value> {
        Ok(self.record(vec![receipt]).await?.remove(0))
    }

    /// Signs receipts and writes them, with one log entry of `kind` holding `input` and the receipts as its output.
    pub async fn record_logged(&self, receipts: Vec<Receipt>, kind: &'static str, input: Value, blobs: Vec<journal::Blob>) -> Result<Vec<Value>> {
        let key = self.key.clone();
        self.db
            .run(move |conn| {
                let at = now();
                let shown = record_receipts(conn, &key, receipts, at)?;
                journal::append(conn, journal::Record::new(kind, at, input, json!({"receipts": shown})).with_blobs(blobs))?;
                Ok(shown)
            })
            .await
    }

    /// Seconds a hotkey polling past its rate waits, None while it is within it.
    pub async fn retry_after(&self, hotkey: &str) -> Result<Option<f64>> {
        let at = now();
        let window = (at / POLL_WINDOW_S).floor() as i64;
        let key = format!("rl:hotkey:{hotkey}:{window}");
        let mut redis = self.redis.clone();
        let count: i64 = redis.incr(&key, 1).await?;
        if count == 1 {
            let _: bool = redis.expire(&key, 2 * POLL_WINDOW_S as i64).await?;
        }
        if count <= 1.max(crate::py::round(self.settings.poll_rate * POLL_WINDOW_S)) {
            return Ok(None);
        }
        Ok(Some(0.01f64.max(round_to((window + 1) as f64 * POLL_WINDOW_S - at, 3))))
    }

    /// Only the first of a run of identical refusals is signed into the log.
    pub async fn first_refusal(&self, hotkey: &str, code: &str) -> Result<bool> {
        let key = format!("refused:{hotkey}:{code}:{}", (now() / MINUTE).floor() as i64);
        let set: Option<String> = redis::cmd("SET").arg(&key).arg("1").arg("NX").arg("EX").arg(2 * MINUTE as i64).query_async(&mut self.redis.clone()).await?;
        Ok(set.is_some())
    }

    /// Whether one more refusal of this hotkey's completions or abandons this minute is logged: the first
    /// `REFUSALS_LOGGED_PER_MINUTE` are, so a relay's witness of any of them finds it in the log.
    pub async fn refusal_to_log(&self, hotkey: &str) -> Result<bool> {
        self.per_minute("refusals", hotkey, REFUSALS_LOGGED_PER_MINUTE).await
    }

    /// Counts one more `name` for this hotkey this minute; false once it is past `limit`.
    pub async fn per_minute(&self, name: &str, hotkey: &str, limit: i64) -> Result<bool> {
        let key = format!("{name}:{hotkey}:{}", (now() / MINUTE).floor() as i64);
        let mut redis = self.redis.clone();
        let count: i64 = redis.incr(&key, 1).await?;
        if count == 1 {
            let _: bool = redis.expire(&key, 2 * MINUTE as i64).await?;
        }
        Ok(count <= limit)
    }

    /// Log reads have a budget of their own, so a dashboard cannot use up the rest.
    pub async fn read_wait(&self, ip: &str, logs: bool) -> Result<Option<i64>> {
        let at = now();
        let (kind, limit) = if logs { ("logs", self.settings.log_reads_per_minute) } else { ("read", self.settings.reads_per_minute) };
        let key = format!("rl:{kind}:{ip}:{}", (at / MINUTE).floor() as i64);
        let mut redis = self.redis.clone();
        let count: i64 = redis.incr(&key, 1).await?;
        if count == 1 {
            let _: bool = redis.expire(&key, 2 * MINUTE as i64).await?;
        }
        Ok((count > limit).then(|| MINUTE as i64 - (at % MINUTE) as i64))
    }

    /// Seconds an address that kept failing authentication waits before it may write again.
    pub async fn write_wait(&self, ip: &str) -> Result<Option<i64>> {
        let at = now();
        let failed: Option<i64> = self.redis.clone().get(format!("rl:denied:{ip}:{}", (at / MINUTE).floor() as i64)).await?;
        Ok((failed.unwrap_or(0) >= self.settings.failed_writes_per_minute).then(|| MINUTE as i64 - (at % MINUTE) as i64))
    }

    pub async fn write_denied(&self, ip: &str) -> Result<()> {
        let key = format!("rl:denied:{ip}:{}", (now() / MINUTE).floor() as i64);
        let mut redis = self.redis.clone();
        let count: i64 = redis.incr(&key, 1).await?;
        if count == 1 {
            let _: bool = redis.expire(&key, 2 * MINUTE as i64).await?;
        }
        Ok(())
    }

    /// Sets `key` only when it is not set yet, expiring after `seconds`.
    pub async fn set_once(&self, key: &str, value: &str, seconds: i64) -> Result<bool> {
        let set: Option<String> = redis::cmd("SET").arg(key).arg(value).arg("NX").arg("EX").arg(seconds).query_async(&mut self.redis.clone()).await?;
        Ok(set.is_some())
    }

    pub async fn delete(&self, key: &str) -> Result<()> {
        let _: i64 = self.redis.clone().del(key).await?;
        Ok(())
    }
}
