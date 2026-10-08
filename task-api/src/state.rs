//! What every request and the janitor share: settings, Redis, SQLite, the buckets, the chain and the receipt key.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result};
use desearch::canonical::hex;
use desearch::hotkey::Hotkey;
use desearch::r2::{Bucket, Credentials};
use desearch::time::now;
use redis::AsyncCommands;
use serde_json::{json, Value};

use crate::chain::Chain;
use crate::db::{Db, Reader};
use crate::flow::PublishRate;
use crate::logs::Cache;
use crate::py::{canonical, round_to};
use crate::queues::{PublishQueue, TaskQueue, ValidationQueue};
use crate::registry::{addresses, Registry};
use crate::roundlog::{self, Receipt};
use crate::rounds::UPLOAD_GRACE_S;
use crate::seeds::Seeds;
use crate::settings::{RegistryMode, SeedsMode, Settings};
use crate::{db, roundstore};

pub type Redis = redis::aio::ConnectionManager;

pub const POLL_WINDOW_S: f64 = 10.0;
pub const MINUTE: f64 = 60.0;
const DEV_RECEIPT_KEY: &str = "//TaskApi";

pub struct State {
    pub settings: Settings,
    pub redis: Redis,
    pub db: Db,
    pub reader: Reader,
    pub cache: Cache,
    pub storage: Bucket,
    pub pages: Bucket,
    pub registry: Arc<Registry>,
    pub chain: Option<Arc<Chain>>,
    pub admins: HashSet<String>,
    pub seeds: Seeds,
    pub key: Hotkey,
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
    /// Claims that expired before this moment plus one claim's time were held through a restart.
    pub started_at: f64,
}

pub async fn redis_from(url: &str) -> Result<Redis> {
    let client = redis::Client::open(url).with_context(|| format!("TASK_API_REDIS {url:?}"))?;
    let config = redis::aio::ConnectionManagerConfig::new()
        .set_connection_timeout(Some(Duration::from_secs(2)))
        .set_response_timeout(Some(Duration::from_secs(10)))
        .set_number_of_retries(1);
    Ok(redis::aio::ConnectionManager::new_with_config(client, config).await?)
}

impl State {
    pub async fn new(settings: Settings, redis: Redis, storage: Bucket, pages: Bucket) -> Result<State> {
        std::fs::create_dir_all(&settings.data).with_context(|| format!("TASK_API_DATA {:?}", settings.data))?;
        let path = settings.data.join(db::FILE);
        let db = Db::open(&path)?;
        let reader = Reader::open(&path)?;
        let current = db.run(roundstore::latest_revealed).await?;
        let network = match (&settings.registry, &settings.seeds) {
            (RegistryMode::Chain { network, .. }, _) | (_, SeedsMode::Chain { network }) => Some(network),
            _ => None,
        };
        let chain = network.map(|network| Chain::new(network)).transpose()?.map(Arc::new);
        let registry = match &settings.registry {
            RegistryMode::Chain { .. } => Registry::chain(),
            RegistryMode::Local { validator_uris } => Registry::local(addresses(validator_uris)?),
        };
        let seeds = match &settings.seeds {
            SeedsMode::Chain { .. } => Seeds::Chain(chain.clone().context("the seeds need the chain")?),
            SeedsMode::Local { block_seconds, genesis } => Seeds::Local { block_seconds: *block_seconds, genesis: *genesis },
        };
        let mut admins: HashSet<String> = settings.admin_hotkeys.iter().cloned().collect();
        admins.extend(addresses(&settings.admin_uris)?);
        let key = Hotkey::from_uri(if settings.key_uri.is_empty() { DEV_RECEIPT_KEY } else { &settings.key_uri }).context("TASK_API_KEY_URI")?;
        let held = settings.claim_ttl + UPLOAD_GRACE_S;
        Ok(State {
            redis,
            db,
            reader,
            cache: Cache::default(),
            storage,
            pages,
            registry: Arc::new(registry),
            chain,
            admins,
            seeds,
            key,
            crawl: TaskQueue::new("crawl", held),
            embed: TaskQueue::new("embed", held),
            validation: ValidationQueue { active_s: settings.active_s },
            publish: PublishQueue { claim_ttl: settings.publish_ttl, max_tries: settings.publish_tries },
            publish_rate: Mutex::new(PublishRate::new(settings.publish_lag, settings.publish_lag_limit)),
            current: Mutex::new(current),
            open_manifests: Mutex::default(),
            open_listed: Mutex::default(),
            started_at: now(),
            settings,
        })
    }

    /// The buckets the settings name, under one set of credentials.
    pub fn buckets(settings: &Settings) -> Result<(Bucket, Bucket)> {
        let http = desearch::r2::client()?;
        let credentials = Credentials { access_key: settings.r2_access_key.clone(), secret_key: settings.r2_secret_key.clone(), region: "auto".into() };
        let storage = Bucket::new(http.clone(), &settings.r2_endpoint, &settings.bucket, &settings.prefix, credentials.clone(), Default::default())?;
        let pages = Bucket::new(http, &settings.r2_endpoint, &settings.pages_bucket, &settings.pages_prefix, credentials, Default::default())?;
        Ok((storage, pages))
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
        let signed: Vec<(Receipt, Value, String)> = receipts
            .into_iter()
            .map(|receipt| {
                let body = receipt.body();
                let signature = self.sign(&canonical(&body));
                (receipt, body, signature)
            })
            .collect();
        let shown = signed.iter().map(|(_, body, signature)| json!({"body": body, "signature": signature})).collect();
        let served_at = now();
        self.db
            .run(move |conn| {
                for (receipt, _, signature) in &signed {
                    roundlog::record(conn, receipt, signature, served_at)?;
                }
                Ok(())
            })
            .await?;
        Ok(shown)
    }

    pub async fn record_one(&self, receipt: Receipt) -> Result<Value> {
        Ok(self.record(vec![receipt]).await?.remove(0))
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
